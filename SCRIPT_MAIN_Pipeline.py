"""
SCRIPT_MAIN_Pipeline.py — Step-by-step MASt3R-SLAM pipeline with Rerun visualization.

This script runs the full SLAM pipeline in clearly separated, documented steps.
It is equivalent to main.py but written for readability and debugging:
each step prints its inputs, outputs, and status so you can see exactly
where things succeed or fail.

Usage:
    micromamba run -n sfm3r python SCRIPT_MAIN_Pipeline.py --dataset data/normal-apt-tour.MOV
    micromamba run -n sfm3r python SCRIPT_MAIN_Pipeline.py --dataset data/normal-apt-tour.MOV --rerun
    micromamba run -n sfm3r python SCRIPT_MAIN_Pipeline.py --dataset data/normal-apt-tour.MOV --no-viz

Pipeline architecture:
    ┌─────────────┐
    │  Dataset     │  Video/image sequence → (timestamp, img) per frame
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Frame 0:   │  Mono inference → initial 3D pointmap + confidence
    │  INIT mode  │  → becomes first keyframe
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Frame i:   │  Tracker matches frame against last keyframe
    │  TRACKING   │  → decides: new keyframe? relocalization needed?
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Backend    │  Retrieval database finds loop closures
    │  (per frame)│  Factor graph optimizes poses + geometry
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Output     │  Trajectory (.txt), reconstruction (.ply),
    │             │  keyframe images, Rerun recording (.rrd)
    └─────────────┘

Author: Von's MASt3R-SLAM Windows adaptation
Date:   2026-02-18
"""

import argparse
import datetime
import pathlib
import sys
import time
import traceback

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Core imports and GPU setup
# ══════════════════════════════════════════════════════════════════════════════

def step_01_imports():
    """Import core libraries and verify GPU availability.

    Inputs:  None (system environment)
    Outputs: torch, lietorch modules confirmed working
    Side effects:
        - mast3r_slam.__init__ auto-patches lietorch on Windows
          (routes Lie group CUDA ops through CPU to avoid kernel crashes)
    """
    print("=" * 70)
    print("STEP 1: Core imports and GPU setup")
    print("=" * 70)

    import torch
    import lietorch

    print(f"  PyTorch:  {torch.__version__}")
    print(f"  CUDA:     {torch.version.cuda}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU:      {gpu_name} ({total_mem:.1f} GB)")
    else:
        print("  GPU:      NOT AVAILABLE — pipeline will be very slow")
    print(f"  lietorch: OK (Lie groups for PyTorch)")

    import platform
    if platform.system() == "Windows":
        print(f"  Platform: Windows — lietorch CPU workaround active")

    return torch, lietorch


def step_02_slam_imports():
    """Import all MASt3R-SLAM modules.

    Inputs:  None
    Outputs: All pipeline modules loaded
    Verifies:
        - Config system
        - Dataset loader
        - Frame/keyframe shared memory structures
        - MASt3R neural network utilities
        - Tracker, factor graph, retrieval database
        - Rerun visualization (optional)
    """
    print("\n" + "=" * 70)
    print("STEP 2: MASt3R-SLAM module imports")
    print("=" * 70)

    from mast3r_slam.config import load_config, config
    from mast3r_slam.dataloader import load_dataset
    from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
    from mast3r_slam.multiprocess_utils import FakeManager
    from mast3r_slam.mast3r_utils import load_mast3r, load_retriever, mast3r_inference_mono
    from mast3r_slam.tracker import FrameTracker
    from mast3r_slam.global_opt import FactorGraph
    import mast3r_slam.evaluate as evaluate

    # Check visualization backends
    try:
        from mast3r_slam.rerun_viz import RerunVisualizer, WindowMsg
        print(f"  Rerun visualization: AVAILABLE")
        has_rerun = True
    except ImportError:
        has_rerun = False
        print(f"  Rerun visualization: not installed")

    print(f"  All SLAM modules imported OK")
    return has_rerun


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3–5: Configuration, dataset, shared state
# ══════════════════════════════════════════════════════════════════════════════

def step_03_load_config(config_path):
    """Load YAML configuration and auto-adapt subsample for GPU memory.

    Inputs:
        config_path (str): Path to YAML config file (e.g. "config/base.yaml")
    Outputs:
        config (dict): Global configuration dict (modified in-place)
    Side effects:
        - If GPU has <10GB VRAM, subsample is raised to 5
        - If GPU has <14GB VRAM, subsample is raised to 3
        - If GPU has <18GB VRAM, subsample is raised to 2
    """
    print("\n" + "=" * 70)
    print("STEP 3: Load configuration")
    print("=" * 70)

    import torch
    from mast3r_slam.config import load_config, config

    load_config(config_path)
    print(f"  Config file:    {config_path}")
    print(f"  single_thread:  {config.get('single_thread', False)}")
    print(f"  use_calib:      {config.get('use_calib')}")
    print(f"  subsample:      {config['dataset']['subsample']}")

    # Auto-adapt subsample based on GPU VRAM
    device = "cuda:0"
    if torch.cuda.is_available():
        _free, total_mem = torch.cuda.mem_get_info(device)
        total_gb = total_mem / (1024 ** 3)
        cfg_sub = config["dataset"]["subsample"]
        if total_gb < 10:
            auto_sub = max(cfg_sub, 5)
        elif total_gb < 14:
            auto_sub = max(cfg_sub, 3)
        elif total_gb < 18:
            auto_sub = max(cfg_sub, 2)
        else:
            auto_sub = cfg_sub
        if auto_sub != cfg_sub:
            config["dataset"]["subsample"] = auto_sub
            print(f"  [GPU auto-adapt] {total_gb:.1f} GB VRAM → subsample {cfg_sub} → {auto_sub}")
        else:
            print(f"  [GPU auto-adapt] {total_gb:.1f} GB VRAM → subsample stays at {cfg_sub}")

    return config


def step_04_load_dataset(dataset_path, config, angles=None, cam="cam0"):
    """Load video/image dataset and apply subsampling.

    Inputs:
        dataset_path (str): Path to video file or image directory
        config (dict):      Global config (uses config["dataset"]["subsample"])
        angles (list|None): If provided, list of angle suffixes for interleaved mode
        cam (str):          Camera prefix for interleaved mode (e.g. "cam0")
    Outputs:
        dataset:  Iterable yielding (timestamp, img_numpy) per frame
        h, w:     Frame dimensions after resizing (int, int)
    """
    print("\n" + "=" * 70)
    print("STEP 4: Load dataset")
    print("=" * 70)

    if angles:
        from mast3r_slam.interleaved_dataset import InterleavedRGBFiles

        dataset = InterleavedRGBFiles(dataset_path, angles, cam=cam)
        n_total = len(dataset)
        dataset.subsample(config["dataset"]["subsample"])
        h, w = dataset.get_img_shape()[0]
        seq_name = f"{cam}_interleaved_{len(angles)}ang"

        print(dataset.summary())
        print(f"  After sub-{config['dataset']['subsample']}: {len(dataset)} frames")
        print(f"  Frame size:   {h} × {w}")
    else:
        from mast3r_slam.dataloader import load_dataset

        dataset = load_dataset(dataset_path)
        n_total = len(dataset)
        dataset.subsample(config["dataset"]["subsample"])
        h, w = dataset.get_img_shape()[0]
        seq_name = dataset.dataset_path.stem

        print(f"  Source:       {dataset_path}")
        print(f"  Sequence:     {seq_name}")
        print(f"  Total frames: {n_total}")
        print(f"  After sub-{config['dataset']['subsample']}: {len(dataset)} frames")
        print(f"  Frame size:   {h} × {w}")

    return dataset, h, w, seq_name


def step_05_shared_state(h, w):
    """Create shared memory structures for keyframes and pipeline state.

    Inputs:
        h, w (int): Frame height and width
    Outputs:
        manager:    FakeManager (single-thread) — no multiprocessing overhead
        keyframes:  SharedKeyframes — stores keyframe poses, pointmaps, images
        states:     SharedStates — tracks pipeline mode (INIT/TRACKING/RELOC),
                    current frame, optimization task queue, graph edges
    """
    print("\n" + "=" * 70)
    print("STEP 5: Create shared state structures")
    print("=" * 70)

    from mast3r_slam.frame import SharedKeyframes, SharedStates
    from mast3r_slam.multiprocess_utils import FakeManager

    manager = FakeManager()
    max_kf = config.get("keyframes", {}).get("max_keyframes", 512)
    keyframes = SharedKeyframes(manager, h, w, buffer=max_kf)
    states = SharedStates(manager, h, w)

    print(f"  Manager:    FakeManager (single-thread, no IPC overhead)")
    print(f"  Keyframes:  buffer for up to {max_kf} keyframes @ {h}×{w}")
    print(f"  States:     mode=INIT, empty task queue")

    return manager, keyframes, states


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6–8: Load neural network models
# ══════════════════════════════════════════════════════════════════════════════

def step_06_load_model(device):
    """Load the MASt3R neural network (ViT-Large encoder + decoder).

    Inputs:
        device (str): CUDA device, e.g. "cuda:0"
    Outputs:
        model: MASt3R model — used for mono inference, stereo matching,
               and feature extraction. ~1.2 GB on GPU.
    Note:
        This is the heaviest memory allocation. On 8GB GPUs, this alone
        uses ~1.5 GB VRAM. Models are loaded BEFORE visualization to
        prevent OOM when Rerun is also active.
    """
    print("\n" + "=" * 70)
    print("STEP 6: Load MASt3R model")
    print("=" * 70)

    import torch
    from mast3r_slam.mast3r_utils import load_mast3r

    free_before = torch.cuda.mem_get_info(device)[0] / (1024**3) if torch.cuda.is_available() else 0
    model = load_mast3r(device=device)
    free_after = torch.cuda.mem_get_info(device)[0] / (1024**3) if torch.cuda.is_available() else 0

    print(f"  Model loaded on {device}")
    print(f"  VRAM used by model: ~{free_before - free_after:.1f} GB")
    print(f"  VRAM remaining:     ~{free_after:.1f} GB")

    return model


def step_07_load_retriever(model):
    """Load the retrieval database for loop closure detection.

    Inputs:
        model: MASt3R model (provides feature encoder for retrieval)
    Outputs:
        retrieval_database: RetrievalDatabase — maintains a database of
            keyframe features and finds similar keyframes for loop closures.
            Uses cosine similarity with configurable k and min_thresh.
    """
    print("\n" + "=" * 70)
    print("STEP 7: Load retrieval database")
    print("=" * 70)

    from mast3r_slam.mast3r_utils import load_retriever

    retrieval_database = load_retriever(model)
    print(f"  Retrieval database ready")
    print(f"  Will find top-k={config['retrieval']['k']} matches per query")
    print(f"  Min similarity threshold: {config['retrieval']['min_thresh']}")

    return retrieval_database


def step_08_create_pipeline_components(model, keyframes, device):
    """Create tracker and factor graph for the SLAM pipeline.

    Inputs:
        model:     MASt3R model (used by tracker for stereo matching)
        keyframes: SharedKeyframes (tracker reads last keyframe for matching)
        device:    CUDA device string
    Outputs:
        tracker:      FrameTracker — matches each new frame against the last
                      keyframe using MASt3R stereo features, estimates relative
                      pose, decides whether to create a new keyframe.
        factor_graph: FactorGraph — accumulates pairwise constraints between
                      keyframes and optimizes all poses jointly via
                      Gauss-Newton on Sim3 (uncalibrated) or SE3 (calibrated).
    """
    print("\n" + "=" * 70)
    print("STEP 8: Create tracker and factor graph")
    print("=" * 70)

    from mast3r_slam.tracker import FrameTracker
    from mast3r_slam.global_opt import FactorGraph

    K = None  # No calibration for uncalibrated mode
    if config["use_calib"]:
        print(f"  WARNING: Calibrated mode — but no K provided here.")

    tracker = FrameTracker(model, keyframes, device)
    factor_graph = FactorGraph(model, keyframes, K, device)

    print(f"  Tracker:      matches frames against last keyframe")
    print(f"    match_frac_thresh: {config['tracking']['match_frac_thresh']}")
    print(f"    max_iters:         {config['tracking']['max_iters']}")
    print(f"  Factor graph: joint pose optimization")
    print(f"    local_opt iters:   {config['local_opt']['max_iters']}")

    return tracker, factor_graph


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9: Output directory and optional Rerun visualization
# ══════════════════════════════════════════════════════════════════════════════

def step_09_setup_output(seq_name, datetime_now, states, keyframes,
                         use_rerun, save_rrd_only=False):
    """Create per-run output directory and optionally start Rerun viewer.

    Inputs:
        seq_name (str):     Dataset sequence name (e.g. "normal-apt-tour")
        datetime_now (str): Timestamp string for this run (e.g. "2026-02-18_140000")
        states:             SharedStates (passed to Rerun visualizer)
        keyframes:          SharedKeyframes (passed to Rerun visualizer)
        use_rerun (bool):   Whether to start live Rerun visualization
        save_rrd_only (bool): Save .rrd file without launching the viewer
    Outputs:
        output_dir (Path):  e.g. logs/normal-apt-tour/2026-02-18_140000/
        rerun_viz:          RerunVisualizer instance or None
    Directory structure created:
        logs/<seq_name>/<timestamp>/
        ├── <seq_name>.txt          # trajectory (TUM format)
        ├── <seq_name>.ply          # point cloud reconstruction
        ├── <seq_name>.rrd          # Rerun recording (if --rerun or --save-rrd)
        └── keyframes/              # keyframe images
    """
    print("\n" + "=" * 70)
    print("STEP 9: Setup output directory and visualization")
    print("=" * 70)

    output_dir = pathlib.Path("logs") / seq_name / datetime_now
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output dir: {output_dir}")

    rerun_viz = None
    if use_rerun or save_rrd_only:
        from mast3r_slam.rerun_viz import RerunVisualizer
        rrd_path = str(output_dir / f"{seq_name}.rrd")
        live = use_rerun and not save_rrd_only
        rerun_viz = RerunVisualizer(states, keyframes, save_path=rrd_path,
                                    live_viewer=live)
        if live:
            print(f"  Rerun:      ACTIVE (viewer + {rrd_path})")
        else:
            print(f"  Rerun:      FILE ONLY → {rrd_path}  (view later with: rerun {rrd_path})")
    else:
        print(f"  Rerun:      disabled (use --rerun or --save-rrd to enable)")

    return output_dir, rerun_viz


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10: Initialize SLAM (first frame)
# ══════════════════════════════════════════════════════════════════════════════

def step_10_init_first_frame(dataset, model, keyframes, states, device):
    """Process the first frame: mono inference → first keyframe.

    Inputs:
        dataset:   Dataset iterator (reads frame 0)
        model:     MASt3R model (runs mono 3D reconstruction)
        keyframes: SharedKeyframes (first keyframe is appended here)
        states:    SharedStates (mode transitions INIT → TRACKING)
        device:    CUDA device
    Outputs:
        frame:    The first Frame object with 3D pointmap and confidence
    Pipeline flow:
        1. Read (timestamp, img) from dataset[0]
        2. Create Frame with identity Sim3 pose
        3. MASt3R mono inference → (X_canon, C) — 3D point positions + confidence
        4. Update frame's pointmap
        5. Append as keyframe #0
        6. Queue global optimization for keyframe #0
        7. Transition mode: INIT → TRACKING
    """
    print("\n" + "=" * 70)
    print("STEP 10: Initialize first frame (INIT → TRACKING)")
    print("=" * 70)

    import lietorch
    from mast3r_slam.frame import Mode, create_frame
    from mast3r_slam.mast3r_utils import mast3r_inference_mono

    timestamp, img = dataset[0]
    T_WC = lietorch.Sim3.Identity(1, device=device)
    frame = create_frame(0, img, T_WC, img_size=dataset.img_size, device=device)
    print(f"  Frame 0:   timestamp={timestamp:.4f}, shape={frame.img_shape.tolist()}")

    X_init, C_init = mast3r_inference_mono(model, frame)
    frame.update_pointmap(X_init, C_init)
    print(f"  Mono 3D:   X shape={X_init.shape}, C shape={C_init.shape}")
    print(f"  Conf range: [{C_init.min().item():.2f}, {C_init.max().item():.2f}]")

    keyframes.append(frame)
    states.queue_global_optimization(len(keyframes) - 1)
    states.set_mode(Mode.TRACKING)
    states.set_frame(frame)

    print(f"  Keyframe #0 created, mode → TRACKING")
    print(f"  Optimization queue: {list(states.global_optimizer_tasks)}")

    return frame


# ══════════════════════════════════════════════════════════════════════════════
# STEP 11: Main SLAM loop
# ══════════════════════════════════════════════════════════════════════════════

def step_11_run_slam_loop(
    dataset, model, tracker, factor_graph, retrieval_database,
    keyframes, states, rerun_viz, device,
    gpu_mem_profile=False,
    gpu_mem_interval=500,
):
    """Run the main SLAM tracking loop over all remaining frames.

    Inputs:
        dataset:             Remaining frames to process (starts at index 1)
        model:               MASt3R model (for mono inference during relocalization)
        tracker:             FrameTracker (matches each frame against last keyframe)
        factor_graph:        FactorGraph (maintains and optimizes pose graph)
        retrieval_database:  RetrievalDatabase (finds loop closures)
        keyframes:           SharedKeyframes (grows as new keyframes are added)
        states:              SharedStates (tracks mode, current frame, edges)
        rerun_viz:           RerunVisualizer or None (updated each frame)
        device:              CUDA device string

    Per-frame flow:
        TRACKING mode:
            1. tracker.track(frame) → (add_new_kf, match_info, try_reloc)
               - Runs MASt3R stereo matching against last keyframe
               - Estimates relative pose via iterative projection + GN
               - Decides if frame is a new keyframe (match_frac below threshold)
            2. If try_reloc → switch to RELOC mode
            3. If add_new_kf → append keyframe, queue optimization

        RELOC mode:
            1. Mono inference on current frame (independent 3D reconstruction)
            2. Queue relocalization — backend searches retrieval database
               for matching keyframes and attempts loop closure

        Backend (every frame):
            1. Pop optimization task from queue
            2. Query retrieval database for loop closure candidates
            3. Add factors (pairwise constraints) to factor graph
            4. Solve Gauss-Newton optimization on all keyframe poses

    Outputs:
        n_keyframes (int):  Total keyframes created
        elapsed (float):    Total processing time in seconds
    """
    print("\n" + "=" * 70)
    print("STEP 11: Main SLAM loop")
    print("=" * 70)

    import lietorch
    from mast3r_slam.frame import Mode, create_frame
    from mast3r_slam.mast3r_utils import mast3r_inference_mono

    fps_timer = time.time()
    n_reloc = 0
    n_new_kf = 0

    for i in range(1, len(dataset)):
        mode = states.get_mode()
        if mode == Mode.TERMINATED:
            break

        timestamp, img = dataset[i]
        T_WC = states.get_frame().T_WC
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

        add_new_kf = False
        do_slice = gpu_mem_profile and i > 0 and i % gpu_mem_interval == 0

        if mode == Mode.TRACKING:
            if do_slice:
                from mast3r_slam.gpu_mem import sample
                sample("11_before_track")
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if do_slice:
                from mast3r_slam.gpu_mem import sample
                sample("11_after_track")
            if try_reloc:
                states.set_mode(Mode.RELOC)
                n_reloc += 1
            states.set_frame(frame)

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()

        if add_new_kf:
            if len(keyframes) < keyframes.buffer:
                keyframes.append(frame)
                states.queue_global_optimization(len(keyframes) - 1)
                n_new_kf += 1
            else:
                print(f"  [Keyframe buffer full ({keyframes.buffer}), skipping new keyframe]")

        if do_slice:
            from mast3r_slam.gpu_mem import sample
            sample("11_before_backend")
        # --- Backend: retrieval + factor graph optimization ---
        _run_backend(states, keyframes, factor_graph, retrieval_database)
        if do_slice:
            from mast3r_slam.gpu_mem import sample
            sample("11_after_backend")

        # --- Rerun visualization update ---
        if rerun_viz is not None:
            rerun_viz.update(frame_idx=i)

        if gpu_mem_profile and i > 0 and (i % gpu_mem_interval == 0 or i == len(dataset) - 1):
            from mast3r_slam.gpu_mem import sample
            sample(f"11_loop frame={i} kf={len(keyframes)}")

        # --- Progress ---
        if i % 30 == 0:
            elapsed = time.time() - fps_timer
            fps = i / elapsed
            print(f"  Frame {i:4d}/{len(dataset)}  |  "
                  f"KFs: {len(keyframes):3d}  |  "
                  f"FPS: {fps:.1f}  |  "
                  f"Mode: {states.get_mode().name}")

    elapsed = time.time() - fps_timer
    fps = (len(dataset) - 1) / max(elapsed, 0.01)

    print(f"\n  ── Loop complete ──")
    print(f"  Frames processed: {len(dataset)}")
    print(f"  Keyframes:        {len(keyframes)}")
    print(f"  Relocalization:   {n_reloc} attempts")
    print(f"  New keyframes:    {n_new_kf}")
    print(f"  Total time:       {elapsed:.1f}s")
    print(f"  Average FPS:      {fps:.2f}")

    return len(keyframes), elapsed


def _run_backend(states, keyframes, factor_graph, retrieval_database):
    """Backend optimization step (called every frame, same as main.py run_backend).

    Handles:
        - RELOC mode: attempts relocalization via retrieval database
        - Normal mode: pops queued keyframes, finds loop closures,
          adds factors to graph, runs Gauss-Newton solver
    """
    from mast3r_slam.frame import Mode

    mode = states.get_mode()
    if mode == Mode.INIT or states.is_paused():
        return

    if mode == Mode.RELOC:
        frame = states.get_frame()
        success = _relocalization(frame, keyframes, factor_graph, retrieval_database)
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

    # Build factor graph edges for this keyframe
    kf_idx = []
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
        print(f"  Database retrieval {idx}: {lc_inds}")

    kf_idx = list(set(kf_idx) - {idx})
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
            states.global_optimizer_tasks.pop(0)


def _relocalization(frame, keyframes, factor_graph, retrieval_database):
    """Attempt to relocalize using retrieval database (same as main.py)."""
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)
            frame_idx = [n_kf - 1] * len(kf_idx)
            print(f"  RELOCALIZING against kf {n_kf - 1} and {kf_idx}")
            if factor_graph.add_factors(
                frame_idx, kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame, add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("  Relocalization SUCCESS")
                successful = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("  Relocalization FAILED")

        if successful:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful


# ══════════════════════════════════════════════════════════════════════════════
# STEP 12: Save results
# ══════════════════════════════════════════════════════════════════════════════

def step_12_save_results(output_dir, seq_name, dataset, keyframes, C_conf_threshold=1.5):
    """Save trajectory, 3D reconstruction, and keyframe images.

    Inputs:
        output_dir (Path):       Per-run output directory
        seq_name (str):          Sequence name for filenames
        dataset:                 Dataset (provides timestamps, save_results flag)
        keyframes:               SharedKeyframes with optimized poses + pointmaps
        C_conf_threshold (float): Confidence filter for point cloud (default 1.5)
    Outputs (files written):
        <output_dir>/<seq_name>.txt   — Camera trajectory in TUM format:
                                        timestamp tx ty tz qx qy qz qw
        <output_dir>/<seq_name>.ply   — Colored point cloud (PLY binary)
        <output_dir>/keyframes/*.png  — Keyframe RGB images
    """
    print("\n" + "=" * 70)
    print("STEP 12: Save results")
    print("=" * 70)

    import mast3r_slam.evaluate as evaluate

    if not dataset.save_results:
        print(f"  Dataset save_results=False, skipping save.")
        return

    evaluate.save_traj(output_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
    print(f"  Trajectory: {output_dir / (seq_name + '.txt')}")

    evaluate.save_reconstruction(
        output_dir, f"{seq_name}.ply", keyframes, C_conf_threshold,
    )
    print(f"  Point cloud: {output_dir / (seq_name + '.ply')}")

    evaluate.save_keyframes(output_dir / "keyframes", dataset.timestamps, keyframes)
    print(f"  Keyframes:   {output_dir / 'keyframes'}/  ({len(keyframes)} images)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Orchestrate all steps
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(
        description="MASt3R-SLAM step-by-step pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python SCRIPT_MAIN_Pipeline.py --dataset data/normal-apt-tour.MOV
  python SCRIPT_MAIN_Pipeline.py --dataset data/normal-apt-tour.MOV --rerun
  python SCRIPT_MAIN_Pipeline.py --dataset data/normal-apt-tour.MOV --no-viz
        """,
    )
    parser.add_argument("--dataset", required=True, help="Path to video or image directory")
    parser.add_argument("--config", default="config/base.yaml", help="YAML config file")
    parser.add_argument("--rerun", action="store_true", help="Enable Rerun live visualization")
    parser.add_argument("--save-rrd", action="store_true",
                        help="Save .rrd recording without launching the live viewer (less GPU)")
    parser.add_argument("--no-viz", action="store_true", help="Disable all visualization")
    parser.add_argument(
        "--angles",
        default=None,
        help="Comma-separated angle suffixes for interleaved multi-angle mode, "
             "e.g. 'p+0_y+0_r+0,p+0_y+30_r+0,p+30_y+0_r+0,p-30_y+0_r+0'. "
             "When set, --dataset should point to the parent folder containing "
             "{cam}_{angle} subfolders.",
    )
    parser.add_argument(
        "--cam", default="cam0",
        help="Camera prefix for interleaved mode (default: cam0)",
    )
    args = parser.parse_args()

    device = "cuda:0"
    datetime_now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          MASt3R-SLAM — Step-by-Step Pipeline                       ║")
    print(f"║          {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    try:
        # Phase 1: Setup
        torch_mod, lietorch_mod = step_01_imports()
        has_rerun = step_02_slam_imports()
        config = step_03_load_config(args.config)
        # GPU memory profiling (optional)
        gpu_mem_cfg = config.get("gpu_mem", {})
        if gpu_mem_cfg.get("profile", False):
            from mast3r_slam.gpu_mem import enable, set_warn_threshold_gb, set_detailed, reset
            enable(True)
            set_warn_threshold_gb(float(gpu_mem_cfg.get("warn_gb", 21.0)))
            set_detailed(bool(gpu_mem_cfg.get("detailed", False)))
            reset()
            print("  [GMEM] Profiling enabled, warn threshold:", gpu_mem_cfg.get("warn_gb", 21), "GB",
                  ", detailed:", gpu_mem_cfg.get("detailed", False))
        angles = [a.strip() for a in args.angles.split(",")] if args.angles else None
        dataset, h, w, seq_name = step_04_load_dataset(
            args.dataset, config, angles=angles, cam=args.cam
        )
        manager, keyframes, states = step_05_shared_state(h, w)
        if gpu_mem_cfg.get("profile", False):
            from mast3r_slam.gpu_mem import sample
            sample("05_shared_state (keyframe buffer)")

        # Phase 2: Load models (heaviest GPU allocation)
        model = step_06_load_model(device)
        if gpu_mem_cfg.get("profile", False):
            from mast3r_slam.gpu_mem import sample
            sample("06_after_model")
        retrieval_database = step_07_load_retriever(model)
        if gpu_mem_cfg.get("profile", False):
            from mast3r_slam.gpu_mem import sample
            sample("07_after_retriever")
        tracker, factor_graph = step_08_create_pipeline_components(model, keyframes, device)

        # Phase 3: Visualization + output directory
        use_rerun = args.rerun and has_rerun and not args.no_viz
        save_rrd_only = args.save_rrd and has_rerun and not args.no_viz and not use_rerun
        output_dir, rerun_viz = step_09_setup_output(
            seq_name, datetime_now, states, keyframes, use_rerun,
            save_rrd_only=save_rrd_only,
        )

        # Phase 4: Run SLAM
        init_frame = step_10_init_first_frame(dataset, model, keyframes, states, device)
        if gpu_mem_cfg.get("profile", False):
            from mast3r_slam.gpu_mem import sample
            sample("10_after_first_keyframe")
        # Log initial keyframe at timeline t=0 so the first keyframe is visible from the start
        if rerun_viz is not None:
            rerun_viz.update(frame_idx=0)
        n_kf, elapsed = step_11_run_slam_loop(
            dataset, model, tracker, factor_graph, retrieval_database,
            keyframes, states, rerun_viz, device,
            gpu_mem_profile=gpu_mem_cfg.get("profile", False),
            gpu_mem_interval=min(500, max(1, len(dataset) // 20)),
        )

        # Phase 5: Save
        states.set_mode(states.get_mode())  # ensure TERMINATED
        step_12_save_results(output_dir, seq_name, dataset, keyframes)
        if gpu_mem_cfg.get("profile", False):
            from mast3r_slam.gpu_mem import sample, print_summary, print_memory_summary
            sample("12_after_save")
            print_summary()
            if gpu_mem_cfg.get("detailed", False):
                print_memory_summary()

        print("\n" + "═" * 70)
        print(f"  ✓ PIPELINE COMPLETE")
        print(f"  Output:     {output_dir}")
        print(f"  Keyframes:  {n_kf}")
        print(f"  Time:       {elapsed:.1f}s ({(len(dataset)-1)/max(elapsed,0.01):.1f} FPS)")
        print("═" * 70)

    except Exception as e:
        print(f"\n{'!'*70}")
        print(f"  PIPELINE FAILED at: {e}")
        print(f"{'!'*70}")
        traceback.print_exc()
        sys.exit(1)
