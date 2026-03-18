import lietorch
import torch
from mast3r_slam.config import config
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.geometry import (
    constrain_points_to_ray,
)
from mast3r_slam.mast3r_utils import mast3r_match_symmetric
import mast3r_slam_backends


class FactorGraph:
    """
    Pose graph: edges = keyframe pairs with 2D–2D matches.
    Full edge tensors are stored on CPU to limit VRAM; only the active window
    is uploaded to GPU for each solve (local_opt.window_size).
    """

    def __init__(self, model, frames: SharedKeyframes, K=None, device="cuda"):
        self.model = model
        self.frames = frames
        self.device = device
        self.cfg = config["local_opt"]
        self.window_size = self.cfg["window_size"]
        self.K = K

        # Full graph on CPU (saves VRAM; ~4 MB per edge at 384×512)
        self._ii_cpu = torch.as_tensor([], dtype=torch.long, device="cpu")
        self._jj_cpu = torch.as_tensor([], dtype=torch.long, device="cpu")
        self._idx_ii2jj_cpu = torch.as_tensor([], dtype=torch.long, device="cpu")
        self._idx_jj2ii_cpu = torch.as_tensor([], dtype=torch.long, device="cpu")
        self._valid_match_j_cpu = torch.as_tensor([], dtype=torch.bool, device="cpu")
        self._valid_match_i_cpu = torch.as_tensor([], dtype=torch.bool, device="cpu")
        self._Q_ii2jj_cpu = torch.as_tensor([], dtype=torch.float32, device="cpu")
        self._Q_jj2ii_cpu = torch.as_tensor([], dtype=torch.float32, device="cpu")

    def add_factors(self, ii, jj, min_match_frac, is_reloc=False):
        kf_ii = [self.frames[idx] for idx in ii]
        kf_jj = [self.frames[idx] for idx in jj]
        feat_i = torch.cat([kf_i.feat for kf_i in kf_ii])
        feat_j = torch.cat([kf_j.feat for kf_j in kf_jj])
        pos_i = torch.cat([kf_i.pos for kf_i in kf_ii])
        pos_j = torch.cat([kf_j.pos for kf_j in kf_jj])
        shape_i = [kf_i.img_true_shape for kf_i in kf_ii]
        shape_j = [kf_j.img_true_shape for kf_j in kf_jj]

        (
            idx_i2j,
            idx_j2i,
            valid_match_j,
            valid_match_i,
            Qii,
            Qjj,
            Qji,
            Qij,
        ) = mast3r_match_symmetric(
            self.model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
        )

        batch_inds = torch.arange(idx_i2j.shape[0], device=idx_i2j.device)[
            :, None
        ].repeat(1, idx_i2j.shape[1])
        Qj = torch.sqrt(Qii[batch_inds, idx_i2j] * Qji)
        Qi = torch.sqrt(Qjj[batch_inds, idx_j2i] * Qij)

        valid_Qj = Qj > self.cfg["Q_conf"]
        valid_Qi = Qi > self.cfg["Q_conf"]
        valid_j = valid_match_j & valid_Qj
        valid_i = valid_match_i & valid_Qi
        nj = valid_j.shape[1] * valid_j.shape[2]
        ni = valid_i.shape[1] * valid_i.shape[2]
        match_frac_j = valid_j.sum(dim=(1, 2)) / nj
        match_frac_i = valid_i.sum(dim=(1, 2)) / ni

        ii_tensor = torch.as_tensor(ii, device=self.device)
        jj_tensor = torch.as_tensor(jj, device=self.device)

        # NOTE: Saying we need both edge directions to be above thrhreshold to accept either
        invalid_edges = torch.minimum(match_frac_j, match_frac_i) < min_match_frac
        consecutive_edges = ii_tensor == (jj_tensor - 1)
        invalid_edges = (~consecutive_edges) & invalid_edges

        if invalid_edges.any() and is_reloc:
            return False

        valid_edges = ~invalid_edges
        ii_tensor = ii_tensor[valid_edges]
        jj_tensor = jj_tensor[valid_edges]
        idx_i2j = idx_i2j[valid_edges]
        idx_j2i = idx_j2i[valid_edges]
        valid_match_j = valid_match_j[valid_edges]
        valid_match_i = valid_match_i[valid_edges]
        Qj = Qj[valid_edges]
        Qi = Qi[valid_edges]

        # Append to CPU storage (keeps VRAM bounded; only window goes to GPU at solve time)
        self._ii_cpu = torch.cat([self._ii_cpu, ii_tensor.cpu()])
        self._jj_cpu = torch.cat([self._jj_cpu, jj_tensor.cpu()])
        self._idx_ii2jj_cpu = torch.cat([self._idx_ii2jj_cpu, idx_i2j.cpu()])
        self._idx_jj2ii_cpu = torch.cat([self._idx_jj2ii_cpu, idx_j2i.cpu()])
        self._valid_match_j_cpu = torch.cat([self._valid_match_j_cpu, valid_match_j.cpu()])
        self._valid_match_i_cpu = torch.cat([self._valid_match_i_cpu, valid_match_i.cpu()])
        self._Q_ii2jj_cpu = torch.cat([self._Q_ii2jj_cpu, Qj.cpu()])
        self._Q_jj2ii_cpu = torch.cat([self._Q_jj2ii_cpu, Qi.cpu()])

        added_new_edges = valid_edges.sum() > 0
        return added_new_edges

    @property
    def ii(self):
        """For viz / external readers: full graph edge indices (on CPU)."""
        return self._ii_cpu

    @property
    def jj(self):
        """For viz / external readers: full graph edge indices (on CPU)."""
        return self._jj_cpu

    def get_window_edges_gpu(self, n_keyframes: int):
        """
        Return edge tensors for the active window on GPU (for solve).
        Only edges with both endpoints in the last window_size keyframes are included.
        """
        try:
            win = int(self.window_size)
        except Exception:
            win = 0
        if win <= 0 or n_keyframes <= win:
            min_kf = 0
        else:
            min_kf = n_keyframes - win

        keep = (self._ii_cpu >= min_kf) & (self._jj_cpu >= min_kf)
        n_keep = keep.sum().item()
        if n_keep == 0:
            # Return empty tensors with correct dtype/device for downstream
            dev = self.device
            return (
                torch.as_tensor([], dtype=torch.long, device=dev),
                torch.as_tensor([], dtype=torch.long, device=dev),
                torch.as_tensor([], dtype=torch.long, device=dev),
                torch.as_tensor([], dtype=torch.long, device=dev),
                torch.as_tensor([], dtype=torch.bool, device=dev),
                torch.as_tensor([], dtype=torch.bool, device=dev),
                torch.as_tensor([], dtype=torch.float32, device=dev),
                torch.as_tensor([], dtype=torch.float32, device=dev),
            )

        return (
            self._ii_cpu[keep].to(self.device),
            self._jj_cpu[keep].to(self.device),
            self._idx_ii2jj_cpu[keep].to(self.device),
            self._idx_jj2ii_cpu[keep].to(self.device),
            self._valid_match_j_cpu[keep].to(self.device),
            self._valid_match_i_cpu[keep].to(self.device),
            self._Q_ii2jj_cpu[keep].to(self.device),
            self._Q_jj2ii_cpu[keep].to(self.device),
        )

    def get_unique_kf_idx(self):
        return torch.unique(torch.cat([self._ii_cpu, self._jj_cpu]), sorted=True)

    def _prep_two_way_from_edges(self, ii, jj, idx_ii2jj, idx_jj2ii, valid_j, valid_i, Q_ii2jj, Q_jj2ii):
        """Build two-way edge lists from the 8 edge tensors (used by solve with window on GPU)."""
        ii_tw = torch.cat((ii, jj), dim=0)
        jj_tw = torch.cat((jj, ii), dim=0)
        idx_tw = torch.cat((idx_ii2jj, idx_jj2ii), dim=0)
        valid_tw = torch.cat((valid_j, valid_i), dim=0)
        Q_tw = torch.cat((Q_ii2jj, Q_jj2ii), dim=0)
        return ii_tw, jj_tw, idx_tw, valid_tw, Q_tw

    def prep_two_way_edges(self):
        """Full graph two-way (CPU); for code paths that still use full graph."""
        ii = torch.cat((self._ii_cpu, self._jj_cpu), dim=0)
        jj = torch.cat((self._jj_cpu, self._ii_cpu), dim=0)
        idx_ii2jj = torch.cat((self._idx_ii2jj_cpu, self._idx_jj2ii_cpu), dim=0)
        valid_match = torch.cat((self._valid_match_j_cpu, self._valid_match_i_cpu), dim=0)
        Q_ii2jj = torch.cat((self._Q_ii2jj_cpu, self._Q_jj2ii_cpu), dim=0)
        return ii, jj, idx_ii2jj, valid_match, Q_ii2jj

    def get_poses_points(self, unique_kf_idx):
        kfs = [self.frames[idx] for idx in unique_kf_idx]
        Xs = torch.stack([kf.X_canon for kf in kfs])
        T_WCs = lietorch.Sim3(torch.stack([kf.T_WC.data for kf in kfs]))

        Cs = torch.stack([kf.get_average_conf() for kf in kfs])

        return Xs, T_WCs, Cs

    def solve_GN_rays(self):
        pin = self.cfg["pin"]
        n_keyframes = len(self.frames)
        (ii, jj, idx_ii2jj, idx_jj2ii, valid_j, valid_i, Q_ii2jj, Q_jj2ii) = self.get_window_edges_gpu(
            n_keyframes
        )
        unique_kf_idx = torch.unique(torch.cat([ii, jj]), sorted=True)
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)

        ii_tw, jj_tw, idx_ii2jj_tw, valid_match_tw, Q_tw = self._prep_two_way_from_edges(
            ii, jj, idx_ii2jj, idx_jj2ii, valid_j, valid_i, Q_ii2jj, Q_jj2ii
        )

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        max_iter = self.cfg["max_iters"]
        sigma_ray = self.cfg["sigma_ray"]
        sigma_dist = self.cfg["sigma_dist"]
        delta_thresh = self.cfg["delta_norm"]

        pose_data = T_WCs.data[:, 0, :]
        mast3r_slam_backends.gauss_newton_rays(
            pose_data,
            Xs,
            Cs,
            ii_tw,
            jj_tw,
            idx_ii2jj_tw,
            valid_match_tw,
            Q_tw,
            sigma_ray,
            sigma_dist,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )

        # Update the keyframe T_WC
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])

    def solve_GN_calib(self):
        K = self.K
        pin = self.cfg["pin"]
        n_keyframes = len(self.frames)
        (ii, jj, idx_ii2jj, idx_jj2ii, valid_j, valid_i, Q_ii2jj, Q_jj2ii) = self.get_window_edges_gpu(
            n_keyframes
        )
        unique_kf_idx = torch.unique(torch.cat([ii, jj]), sorted=True)
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)

        # Constrain points to ray
        img_size = self.frames[0].img.shape[-2:]
        Xs = constrain_points_to_ray(img_size, Xs, K)

        ii_tw, jj_tw, idx_ii2jj_tw, valid_match_tw, Q_tw = self._prep_two_way_from_edges(
            ii, jj, idx_ii2jj, idx_jj2ii, valid_j, valid_i, Q_ii2jj, Q_jj2ii
        )

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        pixel_border = self.cfg["pixel_border"]
        z_eps = self.cfg["depth_eps"]
        max_iter = self.cfg["max_iters"]
        sigma_pixel = self.cfg["sigma_pixel"]
        sigma_depth = self.cfg["sigma_depth"]
        delta_thresh = self.cfg["delta_norm"]

        pose_data = T_WCs.data[:, 0, :]

        img_size = self.frames[0].img.shape[-2:]
        height, width = img_size

        mast3r_slam_backends.gauss_newton_calib(
            pose_data,
            Xs,
            Cs,
            K,
            ii_tw,
            jj_tw,
            idx_ii2jj_tw,
            valid_match_tw,
            Q_tw,
            height,
            width,
            pixel_border,
            z_eps,
            sigma_pixel,
            sigma_depth,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )

        # Update the keyframe T_WC
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])
