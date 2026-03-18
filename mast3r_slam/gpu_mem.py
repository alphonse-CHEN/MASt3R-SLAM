"""
GPU memory (VRAM) profiling for MASt3R-SLAM.

Samples allocated/reserved and peak memory at labeled pipeline stages,
reports delta from previous sample, optionally prints detailed PyTorch
memory stats or summary. Warns when usage exceeds a threshold (e.g. 21 GB).
"""

import torch
from typing import Optional

# Threshold in GB above which we warn (default 21)
GMEM_WARN_GB = 21.0

# Samples: (label, allocated_gb, reserved_gb, delta_gb)
_samples: list[tuple[str, float, float, float]] = []
_peak_allocated_gb: float = 0.0
_peak_reserved_gb: float = 0.0
_prev_alloc_gb: float = 0.0
_enabled: bool = False
_detailed: bool = False


def _bytes_to_gb(b: int) -> float:
    return b / (1024 ** 3)


def _log_memory_stats():
    """Print one-line PyTorch memory stats (when detailed=True)."""
    if not torch.cuda.is_available():
        return
    try:
        s = torch.cuda.memory_stats()
        alloc = s.get("allocated_bytes.all.current", 0) or torch.cuda.memory_allocated()
        reserved = s.get("reserved_bytes.all.current", 0) or torch.cuda.memory_reserved()
        num_alloc = s.get("num_alloc_retries", 0)
        num_oom = s.get("num_ooms", 0)
        print(f"      [GMEM]   active: {_bytes_to_gb(alloc):.3f} GB  reserved: {_bytes_to_gb(reserved):.3f} GB  "
              f"alloc_retries: {num_alloc}  ooms: {num_oom}")
    except Exception:
        pass


def sample(
    label: str,
    warn_threshold_gb: Optional[float] = None,
    detailed: Optional[bool] = None,
) -> tuple[float, float]:
    """
    Record current GPU memory at a named stage. Returns (allocated_gb, reserved_gb).

    When profiling is enabled, stores (label, alloc, reserved, delta_from_previous).
    If allocated > warn_threshold_gb, prints a warning.
    If detailed=True (or global _detailed), prints one-line PyTorch memory stats.
    """
    global _peak_allocated_gb, _peak_reserved_gb, _prev_alloc_gb
    if not torch.cuda.is_available():
        return 0.0, 0.0

    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    alloc_gb = _bytes_to_gb(alloc)
    res_gb = _bytes_to_gb(reserved)
    delta_gb = alloc_gb - _prev_alloc_gb
    _prev_alloc_gb = alloc_gb

    if _enabled:
        _peak_allocated_gb = max(_peak_allocated_gb, alloc_gb)
        _peak_reserved_gb = max(_peak_reserved_gb, res_gb)
        _samples.append((label, alloc_gb, res_gb, delta_gb))

    thr = warn_threshold_gb if warn_threshold_gb is not None else GMEM_WARN_GB
    if alloc_gb > thr:
        print(f"  [GMEM] WARNING: {label} — allocated {alloc_gb:.2f} GB (>{thr:.0f} GB)")

    if (_detailed or detailed) and _enabled:
        _log_memory_stats()

    return alloc_gb, res_gb


def enable(enabled: bool = True):
    global _enabled
    _enabled = enabled


def set_detailed(detailed: bool = True):
    """When True, sample() also prints one-line PyTorch memory stats."""
    global _detailed
    _detailed = detailed


def set_warn_threshold_gb(gb: float):
    global GMEM_WARN_GB
    GMEM_WARN_GB = gb


def get_peak_gb() -> tuple[float, float]:
    """Return (peak_allocated_gb, peak_reserved_gb) since last reset."""
    return _peak_allocated_gb, _peak_reserved_gb


def reset():
    """Clear samples and peak counters (e.g. at start of a run)."""
    global _samples, _peak_allocated_gb, _peak_reserved_gb, _prev_alloc_gb
    _samples = []
    _peak_allocated_gb = 0.0
    _peak_reserved_gb = 0.0
    _prev_alloc_gb = 0.0


def print_summary():
    """Print a table: label, allocated, reserved, delta from previous."""
    if not _samples:
        return
    print("\n  ─── GMEM profile (delta = change since previous sample) ───")
    for label, alloc_gb, res_gb, delta_gb in _samples:
        delta_str = f"{delta_gb:+.2f}" if delta_gb != 0 else " 0.00"
        print(f"    {label:42s}  alloc: {alloc_gb:5.2f} GB  reserved: {res_gb:5.2f} GB   delta: {delta_str} GB")
    p_alloc, p_res = get_peak_gb()
    print(f"    {'PEAK':42s}  alloc: {p_alloc:5.2f} GB  reserved: {p_res:5.2f} GB")
    print("  ─────────────────────────────────────────────────────────────\n")


def print_memory_summary():
    """Print PyTorch's full memory summary (call once at end for deep dive)."""
    if not torch.cuda.is_available():
        return
    try:
        print("\n  ─── PyTorch memory summary (abbreviated) ───")
        print(torch.cuda.memory_summary(abbreviated=True))
        print("  ───────────────────────────────────────────\n")
    except Exception as e:
        print(f"  [GMEM] memory_summary not available: {e}\n")
