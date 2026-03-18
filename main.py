import argparse
import datetime
import pathlib
import sys
import time
import cv2
import lietorch
import torch
import tqdm
import yaml
from mast3r_slam.global_opt import FactorGraph

from mast3r_slam.config import load_config, config
from mast3r_slam.dataloader import Intrinsics, load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg, FakeManager
from mast3r_slam.tracker import FrameTracker

# Visualization backends: Rerun (preferred) or in3d (legacy)
try:
    from mast3r_slam.rerun_viz import RerunVisualizer, WindowMsg
    HAS_RERUN = True
except ImportError:
    HAS_RERUN = False

try:
    from mast3r_slam.visualization import WindowMsg as _LegacyWindowMsg, run_visualization
    HAS_IN3D = True
except ImportError:
    HAS_IN3D = False

if not HAS_RERUN and not HAS_IN3D:
    import dataclasses
    @dataclasses.dataclass
    class WindowMsg:
        is_terminated: bool = False
        is_paused: bool = False
        next: bool = False
        C_conf_threshold: float = 1.5

    def run_visualization(*args, **kwargs):
        raise RuntimeError(
            "No visualization backend available. "
            "Install rerun-sdk (pip install rerun-sdk) or in3d."
        )

if not HAS_RERUN:
    # Use legacy WindowMsg if rerun is not available
    if HAS_IN3D:
        WindowMsg = _LegacyWindowMsg

import torch.multiprocessing as mp


def relocalization(frame, keyframes, factor_graph, retrieval_database):
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx and len(keyframes) < keyframes.buffer:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(states, keyframes):
    mode = states.get_mode()
    if mode == Mode.INIT or states.is_paused():
        return
    if mode == Mode.RELOC:
        frame = states.get_frame()
        success = relocalization(frame, keyframes, factor_graph, retrieval_database)
        if success:
            states.set_mode(Mode.TRACKING)
        states.dequeue_reloc()
        return
    idx = -1
    with states.lock:
        if len(states.global_optimizer_tasks) > 0:
            idx = states.global_optimizer_tasks[0]
    if idx == -1:
        return
    # Graph Construction
    kf_idx = []
    # k to previous consecutive keyframes
    n_consec = 1
    for j in range(min(n_consec, idx)):
        kf_idx.append(idx - 1 - j)
    frame = keyframes[idx]
    retrieval_inds = retrieval_database.update(
        frame,
        add_after_query=True,
        k=config["retrieval"]["k"],
        min_thresh=config["retrieval"]["min_thresh"],
    )
    kf_idx += retrieval_inds

    lc_inds = set(retrieval_inds)
    lc_inds.discard(idx - 1)
    if len(lc_inds) > 0:
        print("Database retrieval", idx, ": ", lc_inds)

    kf_idx = set(kf_idx)  # Remove duplicates by using set
    kf_idx.discard(idx)  # Remove current kf idx if included
    kf_idx = list(kf_idx)  # convert to list
    frame_idx = [idx] * len(kf_idx)
    if kf_idx:
        factor_graph.add_factors(
            kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
        )

    with states.lock:
        states.edges_ii[:] = factor_graph.ii.cpu().tolist()
        states.edges_jj[:] = factor_graph.jj.cpu().tolist()

    if config["use_calib"]:
        factor_graph.solve_GN_calib()
    else:
        factor_graph.solve_GN_rays()

    with states.lock:
        if len(states.global_optimizer_tasks) > 0:
            idx = states.global_optimizer_tasks.pop(0)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True

    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="Use Rerun visualizer (works in single-thread mode)")
    parser.add_argument("--calib", default="")

    args = parser.parse_args()

    load_config(args.config)

    # --- Auto-adapt subsample based on available GPU memory ---
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        total_gb = total_mem / (1024 ** 3)
        cfg_sub = config["dataset"]["subsample"]
        if total_gb < 10:        # 8 GB class (e.g. RTX 4060 Laptop)
            auto_sub = max(cfg_sub, 5)
        elif total_gb < 14:      # 12 GB class (e.g. RTX 4070)
            auto_sub = max(cfg_sub, 3)
        elif total_gb < 18:      # 16 GB class (e.g. RTX 4080)
            auto_sub = max(cfg_sub, 2)
        else:                    # 24 GB+ (RTX 4090, A6000, etc.)
            auto_sub = cfg_sub   # use config as-is
        if auto_sub != cfg_sub:
            print(f"[GPU auto-adapt] {total_gb:.1f} GB VRAM detected — "
                  f"subsample {cfg_sub} → {auto_sub}")
            config["dataset"]["subsample"] = auto_sub

    print(args.dataset)
    print(config)

    single_thread = config.get("single_thread", False)
    use_rerun = args.rerun and HAS_RERUN
    if single_thread:
        manager = FakeManager()
        if not use_rerun:
            args.no_viz = True  # legacy in3d visualization requires multiprocessing
    else:
        mp.set_sharing_strategy('file_system')
        mp.set_start_method("spawn")
        manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    dataset = load_dataset(args.dataset)
    dataset.subsample(config["dataset"]["subsample"])
    h, w = dataset.get_img_shape()[0]

    if args.calib:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        dataset.use_calibration = True
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"],
        )

    max_kf = config.get("keyframes", {}).get("max_keyframes", 512)
    keyframes = SharedKeyframes(manager, h, w, buffer=max_kf)
    states = SharedStates(manager, h, w)

    # Load models FIRST (heavy GPU memory), then start visualization
    model = load_mast3r(device=device)
    if not single_thread:
        model.share_memory()

    # Create a per-run output directory: logs/<seq_name>/<timestamp>/
    seq_name = dataset.dataset_path.stem
    output_dir = pathlib.Path("logs") / seq_name / datetime_now
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    rerun_viz = None
    if use_rerun and not args.no_viz:
        rrd_path = str(output_dir / f"{seq_name}.rrd")
        rerun_viz = RerunVisualizer(states, keyframes, save_path=rrd_path)
        print("Rerun visualization active")
    elif not args.no_viz:
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main),
        )
        viz.start()

    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)

    # No need to remove old results — each run gets its own timestamped folder

    tracker = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()

    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    i = 0
    fps_timer = time.time()

    frames = []

    while True:
        mode = states.get_mode()
        msg = try_get_msg(viz2main)
        last_msg = msg if msg is not None else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        if i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i]
        if save_frames:
            frames.append(img)

        # get frames last camera pose
        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            i += 1
            continue

        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
        else:
            raise Exception("Invalid mode")

        if add_new_kf and len(keyframes) < keyframes.buffer:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)

        run_backend(states, keyframes)

        # Update Rerun visualization (inline, no multiprocessing needed)
        if rerun_viz is not None:
            rerun_viz.update(frame_idx=i)

        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1

    if dataset.save_results:
        eval.save_traj(output_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
        eval.save_reconstruction(
            output_dir,
            f"{seq_name}.ply",
            keyframes,
            last_msg.C_conf_threshold,
        )
        eval.save_keyframes(
            output_dir / "keyframes", dataset.timestamps, keyframes
        )
    if save_frames:
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True, parents=True)
        for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
            frame = (frame * 255).clip(0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frames_dir / f"{i}.png"), frame)

    print("done")
    if not args.no_viz and not use_rerun:
        viz.join()
