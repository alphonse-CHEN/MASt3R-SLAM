"""
Rerun-based visualization for MASt3R-SLAM.

Replaces the in3d/OpenGL visualizer with Rerun, providing:
- 3D point cloud visualization (keyframe maps)
- Camera frustum display (current + keyframes)
- Keyframe connectivity graph (edges)
- Current frame + keyframe image panels
- Interactive confidence threshold control

Usage:
    from mast3r_slam.rerun_viz import RerunVisualizer
    viz = RerunVisualizer(states, keyframes)
    viz.update()  # call each frame from main loop
"""

import dataclasses
import numpy as np
import torch

try:
    import rerun as rr
    import rerun.blueprint as rrb

    HAS_RERUN = True
except ImportError:
    HAS_RERUN = False

import lietorch
from mast3r_slam.config import config
from mast3r_slam.frame import Mode
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.geometry import get_pixel_coords


@dataclasses.dataclass
class WindowMsg:
    """Message from visualizer to main loop (matches original visualization.py)."""

    is_terminated: bool = False
    is_paused: bool = False
    next: bool = False
    C_conf_threshold: float = 1.5


class RerunVisualizer:
    """Rerun-based SLAM visualizer replacing the in3d/OpenGL Window."""

    def __init__(
        self,
        states,
        keyframes,
        C_conf_threshold: float = 2.0,
        max_points_per_cloud: int = 50_000,
        app_id: str = "MASt3R-SLAM",
        save_path: str | None = None,
        live_viewer: bool = True,
    ):
        """Args:
            C_conf_threshold: Only log points with confidence > this (higher = fewer points, smaller recording).
            max_points_per_cloud: Cap points per keyframe/current cloud to limit recording size.
            save_path: If set, write a .rrd file to this path.
            live_viewer: If True, spawn the Rerun viewer for live display.
                         If False, only write to save_path (no viewer, no GPU overhead).
        """
        if not HAS_RERUN:
            raise ImportError(
                "rerun-sdk is required for visualization. "
                "Install with: pip install rerun-sdk"
            )

        self.states = states
        self.keyframes = keyframes
        self.C_conf_threshold = C_conf_threshold
        self.max_points_per_cloud = max_points_per_cloud
        self.logged_kf_ids = set()  # track which keyframes have been logged
        self.last_n_keyframes = 0
        self.dP_dz = None  # cache for calibrated projection

        rr.init(app_id)

        if live_viewer:
            import subprocess, os, time
            if os.name == 'nt':
                subprocess.run(["taskkill", "/f", "/im", "rerun.exe"],
                               capture_output=True)
            else:
                subprocess.run(["pkill", "-f", "rerun"], capture_output=True)
            time.sleep(1)
            rr.spawn(connect=True)

        if save_path is not None:
            import pathlib
            pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        if live_viewer and save_path is not None:
            rec = rr.get_global_data_recording()
            rec.set_sinks(
                rr.GrpcSink(),
                rr.FileSink(save_path),
            )
            print(f"Rerun recording will be saved to: {save_path}")
        elif save_path is not None:
            rr.save(save_path)
            print(f"Rerun recording (file only, no viewer): {save_path}")

        # Set up coordinate system (right-hand, Y-up like the original viewer)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

        # Log a blueprint for the layout
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="3D Scene",
                    origin="/world",
                ),
                rrb.Vertical(
                    rrb.Spatial2DView(
                        name="Current Frame",
                        origin="/images/current",
                    ),
                    rrb.Spatial2DView(
                        name="Keyframe",
                        origin="/images/keyframe",
                    ),
                ),
                column_shares=[3, 1],
            ),
        )
        rr.send_blueprint(blueprint)

    def update(self, frame_idx: int = 0):
        """Update visualization with current SLAM state. Call each frame."""
        try:
            self._update_impl(frame_idx)
        except Exception as e:
            print(f"[RerunViz] update error at frame {frame_idx}: {e}")
            import traceback; traceback.print_exc()

    def _update_impl(self, frame_idx: int):
        rr.set_time("frame", sequence=frame_idx)

        mode = self.states.get_mode()
        if mode == Mode.INIT:
            return

        # --- Current frame ---
        curr_frame = self.states.get_frame()
        self._log_current_frame(curr_frame)

        # --- Keyframes ---
        with self.keyframes.lock:
            n_kf = len(self.keyframes)
            dirty_idx = self.keyframes.get_dirty_idx()

        # Log new/dirty keyframes
        for kf_idx in dirty_idx:
            kf = self.keyframes[kf_idx]
            self._log_keyframe(kf_idx, kf)

        # Log the latest keyframe image
        if n_kf > 0:
            last_kf = self.keyframes[n_kf - 1]
            img_np = last_kf.uimg.numpy()
            img_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)
            rr.log("/images/keyframe", rr.Image(img_uint8))

        # --- Keyframe edges ---
        self._log_edges()

        # --- Current point cloud ---
        if mode != Mode.INIT:
            self._log_current_pointcloud(curr_frame)

        self.last_n_keyframes = n_kf

    def _log_current_frame(self, frame):
        """Log current camera pose, image."""
        # Image
        img_np = frame.uimg.numpy()
        img_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)
        rr.log("/images/current", rr.Image(img_uint8))

        # Camera pose in 3D
        T_WC = as_SE3(frame.T_WC).cpu()
        mat = T_WC.matrix().numpy().squeeze()  # 4x4
        translation = mat[:3, 3]
        rotation_mat = mat[:3, :3]

        rr.log(
            "/world/current_camera",
            rr.Transform3D(
                translation=translation,
                mat3x3=rotation_mat,
            ),
        )

        # Camera frustum indicator
        h, w = frame.img_shape.flatten().tolist()
        # Use a simple pinhole for visualization (approximate focal length)
        focal = max(h, w) * 0.8
        rr.log(
            "/world/current_camera/pinhole",
            rr.Pinhole(
                focal_length=focal,
                width=w,
                height=h,
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.05,
                color=[0, 255, 0],
            ),
        )

    def _log_keyframe(self, kf_idx: int, keyframe):
        """Log a keyframe's point cloud and camera frustum."""
        h, w = keyframe.img_shape.flatten().tolist()

        # --- Camera frustum ---
        T_WC = as_SE3(keyframe.T_WC).cpu()
        mat = T_WC.matrix().numpy().squeeze()  # 4x4
        translation = mat[:3, 3]
        rotation_mat = mat[:3, :3]

        entity = f"/world/keyframes/kf_{kf_idx}"

        rr.log(
            entity,
            rr.Transform3D(
                translation=translation,
                mat3x3=rotation_mat,
            ),
        )

        focal = max(h, w) * 0.8
        rr.log(
            f"{entity}/pinhole",
            rr.Pinhole(
                focal_length=focal,
                width=w,
                height=h,
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.03,
                color=[255, 80, 80],
            ),
        )

        # --- Point cloud ---
        X = self._frame_X(keyframe)  # (H*W, 3) numpy
        C = keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
        colors = (keyframe.uimg.numpy() * 255).clip(0, 255).astype(np.uint8).reshape(-1, 3)

        # Transform points to world frame
        T_WC_sim3 = keyframe.T_WC
        pW = T_WC_sim3.act(
            torch.from_numpy(X).to(T_WC_sim3.data.device)
        ).cpu().numpy().reshape(-1, 3)

        # Filter by confidence and cap count to limit recording size
        valid = C > self.C_conf_threshold
        n_valid = int(valid.sum())
        if n_valid > 0:
            pW_v = pW[valid]
            colors_v = colors[valid]
            if n_valid > self.max_points_per_cloud:
                step = n_valid / self.max_points_per_cloud
                idx = (np.arange(self.max_points_per_cloud) * step).astype(np.int64)
                pW_v = pW_v[idx]
                colors_v = colors_v[idx]
            rr.log(
                f"/world/pointclouds/kf_{kf_idx}",
                rr.Points3D(
                    positions=pW_v,
                    colors=colors_v,
                    radii=0.003,
                ),
            )

        self.logged_kf_ids.add(kf_idx)

    def _log_current_pointcloud(self, frame):
        """Log the current frame's point cloud (colored by depth)."""
        if frame.X_canon is None or frame.C is None:
            return

        X = self._frame_X(frame)  # (H*W, 3) numpy
        C = frame.C.cpu().numpy().astype(np.float32).reshape(-1)

        # Transform to world
        T_WC = frame.T_WC
        pW = T_WC.act(
            torch.from_numpy(X).to(T_WC.data.device)
        ).cpu().numpy().reshape(-1, 3)

        # Color by depth (turbo colormap approximation)
        depth = X[..., 2].reshape(-1)
        valid = C > self.C_conf_threshold
        n_valid = int(valid.sum())
        if n_valid == 0:
            return

        pW_v = pW[valid]
        depth_valid = depth[valid]
        d_min, d_max = np.percentile(depth_valid, [5, 95])
        d_range = max(d_max - d_min, 1e-6)
        depth_norm = ((depth_valid - d_min) / d_range).clip(0, 1)

        # Simple turbo-ish colormap: blue → cyan → green → yellow → red
        r = (255 * np.clip(4 * depth_norm - 1.5, 0, 1)).astype(np.uint8)
        g = (255 * np.clip(np.minimum(4 * depth_norm, -4 * depth_norm + 4), 0, 1)).astype(np.uint8)
        b = (255 * np.clip(1.5 - 4 * depth_norm, 0, 1)).astype(np.uint8)
        depth_colors = np.stack([r, g, b], axis=-1)

        # Cap point count to limit recording size
        if n_valid > self.max_points_per_cloud:
            step = n_valid / self.max_points_per_cloud
            idx = (np.arange(self.max_points_per_cloud) * step).astype(np.int64)
            pW_v = pW_v[idx]
            depth_colors = depth_colors[idx]

        rr.log(
            "/world/current_points",
            rr.Points3D(
                positions=pW_v,
                colors=depth_colors,
                radii=0.002,
            ),
        )

    def _log_edges(self):
        """Log keyframe connectivity graph as line segments."""
        with self.states.lock:
            ii = list(self.states.edges_ii)
            jj = list(self.states.edges_jj)

        if not ii or not jj:
            return

        ii_t = torch.tensor(ii, dtype=torch.long)
        jj_t = torch.tensor(jj, dtype=torch.long)

        with self.keyframes.lock:
            T_WCi = lietorch.Sim3(self.keyframes.T_WC[ii_t, 0])
            T_WCj = lietorch.Sim3(self.keyframes.T_WC[jj_t, 0])

        t_i = T_WCi.matrix()[:, :3, 3].cpu().numpy()  # (N, 3)
        t_j = T_WCj.matrix()[:, :3, 3].cpu().numpy()  # (N, 3)

        # Build line strips: each edge is a 2-point strip
        strips = []
        for k in range(len(ii)):
            strips.append([t_i[k].tolist(), t_j[k].tolist()])

        rr.log(
            "/world/edges",
            rr.LineStrips3D(
                strips,
                colors=[0, 255, 0],
                radii=0.001,
            ),
        )

    def _frame_X(self, frame):
        """Get frame points in camera coordinates, handling calibrated/uncalibrated cases."""
        if config["use_calib"]:
            Xs = frame.X_canon[None]
            if self.dP_dz is None:
                device = Xs.device
                dtype = Xs.dtype
                img_size = frame.img_shape.flatten()[:2]
                K = frame.K
                p = get_pixel_coords(
                    Xs.shape[0], img_size, device=device, dtype=dtype
                ).view(*Xs.shape[:-1], 2)
                tmp1 = (p[..., 0] - K[0, 2]) / K[0, 0]
                tmp2 = (p[..., 1] - K[1, 2]) / K[1, 1]
                self.dP_dz = torch.empty(
                    p.shape[:-1] + (3, 1), device=device, dtype=dtype
                )
                self.dP_dz[..., 0, 0] = tmp1
                self.dP_dz[..., 1, 0] = tmp2
                self.dP_dz[..., 2, 0] = 1.0
                self.dP_dz = self.dP_dz[..., 0].cpu().numpy().astype(np.float32)
            return (Xs[..., 2:3].cpu().numpy().astype(np.float32) * self.dP_dz)[0]
        return frame.X_canon.cpu().numpy().astype(np.float32)
