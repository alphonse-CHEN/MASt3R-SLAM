# MASt3R-SLAM Pipeline — Usage Guide

## Prerequisites

- **Micromamba environment**: `sfm3rv2` with all dependencies installed
- **GPU**: NVIDIA GPU with CUDA (tested on RTX 4090 48 GB)
- **Model checkpoint**: `checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`
- **Retrieval checkpoint**: `checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth`

All commands below assume you are in the project root:

```powershell
cd d:\MASt3R-SLAM
```

---

## Quick Start

### Single image folder

```powershell
micromamba run -n sfm3rv2 python SCRIPT_MAIN_Pipeline.py --dataset "path\to\image_folder"
```

### Video file

```powershell
micromamba run -n sfm3rv2 python SCRIPT_MAIN_Pipeline.py --dataset "path\to\video.mp4"
```

Supported video formats: `.mp4`, `.avi`, `.MOV`, `.mov`

Supported image formats: `.png`, `.jpg`, `.jpeg`, `.bmp`

---

## Command-Line Options

| Flag | Description |
|------|-------------|
| `--dataset PATH` | **(Required)** Path to a video file, image folder, or parent folder (with `--angles`) |
| `--config PATH` | YAML config file (default: `config/base.yaml`) |
| `--rerun` | Launch live Rerun viewer **and** save `.rrd` file |
| `--save-rrd` | Save `.rrd` file **without** launching the viewer (saves GPU memory) |
| `--no-viz` | Disable all visualization (fastest, lowest memory) |
| `--angles LIST` | Comma-separated angle suffixes for multi-angle interleaved mode |
| `--cam PREFIX` | Camera prefix for multi-angle mode (default: `cam0`) |

### Visualization modes

```
--rerun          Live viewer + .rrd file saved    (most GPU)
--save-rrd       .rrd file saved, no viewer       (moderate)
(neither)        No visualization at all           (least GPU)
--no-viz         Force disable even if others set
```

To view a saved `.rrd` after the run:

```powershell
rerun path\to\output.rrd
```

---

## Examples

### 1. Single angle, no visualization (fastest)

```powershell
micromamba run -n sfm3rv2 python SCRIPT_MAIN_Pipeline.py ^
    --dataset "E:\Demo0316\...\staged_images\cam0_p+0_y+0_r+0"
```

### 2. Single angle, save .rrd for later viewing

```powershell
micromamba run -n sfm3rv2 python SCRIPT_MAIN_Pipeline.py ^
    --dataset "E:\Demo0316\...\staged_images\cam0_p+0_y+0_r+0" ^
    --save-rrd
```

### 3. Single angle, live Rerun viewer

```powershell
micromamba run -n sfm3rv2 python SCRIPT_MAIN_Pipeline.py ^
    --dataset "E:\Demo0316\...\staged_images\cam0_p+0_y+0_r+0" ^
    --rerun
```

### 4. Multi-angle interleaved mode

When your data has multiple angle subfolders under a parent directory:

```
staged_images/
├── cam0_p+0_y+0_r+0/       ← center
├── cam0_p+0_y+30_r+0/      ← yaw +30°
├── cam0_p+30_y+0_r+0/      ← pitch +30°
└── cam0_p-30_y+0_r+0/      ← pitch -30°
```

Point `--dataset` to the **parent** folder and list the angles:

```powershell
micromamba run -n sfm3rv2 python SCRIPT_MAIN_Pipeline.py ^
    --dataset "E:\Demo0316\...\staged_images" ^
    --angles "p+0_y+0_r+0,p+0_y+30_r+0,p+30_y+0_r+0,p-30_y+0_r+0" ^
    --cam cam0 ^
    --save-rrd
```

The pipeline will:
1. Find all `{cam}_{angle}` subfolders (e.g. `cam0_p+0_y+0_r+0`)
2. Match images across folders by their shared timestamp key
3. Automatically reorder angles to minimize the angular gap between consecutive frames
4. Interleave: for each timestamp, cycle through all angles before advancing

**Note**: Multi-angle mode multiplies the frame count (597 images x 4 angles = 2,388 frames). This uses significantly more GPU memory. If GMEM is a concern, run single-angle instead.

### 5. Using a different camera

```powershell
micromamba run -n sfm3rv2 python SCRIPT_MAIN_Pipeline.py ^
    --dataset "E:\Demo0316\...\staged_images" ^
    --angles "p+0_y+0_r+0,p+0_y-30_r+0,p+30_y+0_r+0,p-30_y+0_r+0" ^
    --cam cam1
```

---

## Output Structure

Each run creates a timestamped folder under `logs/`:

```
logs/<seq_name>/<YYYY-MM-DD_HHMMSS>/
├── <seq_name>.txt          # camera trajectory (TUM format)
├── <seq_name>.ply          # 3D point cloud reconstruction
├── <seq_name>.rrd          # Rerun recording (if --rerun or --save-rrd)
└── keyframes/              # saved keyframe images
```

- **Trajectory** (`.txt`): One line per keyframe — `timestamp tx ty tz qx qy qz qw`
- **Point cloud** (`.ply`): Colored 3D points from all keyframes (filtered by confidence)
- **Rerun recording** (`.rrd`): Full visualization with camera frustums, trajectory edges, point clouds

---

## Configuration (`config/base.yaml`)

Key parameters you may want to tune:

### GPU memory

```yaml
keyframes:
  max_keyframes: 256      # Cap on number of keyframes (fewer = less VRAM)

gpu_mem:
  profile: true           # Print VRAM usage at each pipeline stage
  warn_gb: 21.0           # Warn when allocated VRAM exceeds this (GB)
  detailed: false          # true = print PyTorch memory summary at end
```

### Dataset

```yaml
dataset:
  subsample: 1            # Process every Nth frame (2 = skip every other frame)
```

### Tracking

```yaml
tracking:
  match_frac_thresh: 0.333  # New keyframe when match fraction drops below this
                             # Higher = fewer keyframes, lower = more keyframes
```

### Backend optimization

```yaml
local_opt:
  window_size: 80          # Sliding window: only optimize over the last N keyframes
                            # Larger = better accuracy, more VRAM
  min_match_frac: 0.1      # Min match fraction to add an edge in the factor graph
```

### Retrieval (loop closure)

```yaml
retrieval:
  k: 3                    # Number of loop closure candidates per keyframe
  min_thresh: 5e-3         # Minimum similarity to consider a match
```

---

## GPU Memory Tips

The main VRAM consumers, roughly in order:

1. **MASt3R model** — ~2.7 GB (fixed)
2. **Keyframe buffer** — ~10-15 MB per keyframe (scales with `max_keyframes`)
3. **Factor graph** — stored on CPU, only active window on GPU during solve
4. **Inference temporaries** — spikes during each forward pass

If you run out of VRAM:

- **First**: reduce `max_keyframes` (e.g. 128 instead of 256)
- **Second**: increase `dataset.subsample` (e.g. 2 or 4) to process fewer frames
- **Third**: use `--save-rrd` instead of `--rerun` (viewer process uses extra GPU)
- **Fourth**: use `--no-viz` to skip visualization entirely
- **Fifth**: reduce `local_opt.window_size` (e.g. 40 instead of 80)

---

## Filename Convention (Multi-Angle Mode)

Image filenames must follow this pattern:

```
{frame_idx}_{timestamp}_{timestamp}_{cam}_{angle}.ext
```

Example: `000000_72966704786293_72966704786293_cam0_p+0_y+30_r+0.jpg`

- `{frame_idx}`: Zero-padded frame number (e.g. `000000`)
- `{timestamp}`: Capture timestamp in nanoseconds (appears twice)
- `{cam}_{angle}`: Camera and orientation identifier

The first three fields (`frame_idx_timestamp_timestamp`) are used as the shared key to match images across angle folders. All angle folders must have the same set of timestamps.

Angle strings follow the format `p{pitch}_y{yaw}_r{roll}` in degrees (e.g. `p+30_y+0_r+0` = pitch +30°, yaw 0°, roll 0°).
