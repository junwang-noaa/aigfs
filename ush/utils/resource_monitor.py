#!/usr/bin/env python3
'''
Description: Background sampler of host (CPU) memory for the forecast process tree,
             GPU memory from JAX/XLA and nvidia-smi, and the Slurm job cgroup peak.
             Used by run_graphcast.py to report the memory needed by the job.
'''
import os
import queue
import subprocess
import threading
from time import time

GB = 1024 ** 3


def _read_kb(path, key):
    """Value in bytes of a 'Key:   123 kB' line in a /proc file, or 0."""
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(key):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _children_map():
    children = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat") as f:
                # ppid is the 2nd field after the parenthesized command name
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(pid))
    return children


def _descendants(pid):
    children = _children_map()
    out, stack = [], list(children.get(pid, []))
    while stack:
        p = stack.pop()
        out.append(p)
        stack.extend(children.get(p, []))
    return out


def _pss(pid):
    # PSS splits copy-on-write pages shared by the fork server and its workers, so
    # summing it over the process tree gives the real footprint.
    return _read_kb(f"/proc/{pid}/smaps_rollup", "Pss:")


def _job_cgroup():
    """Slurm job-level cgroup v2 directory (job_<id>), else this process's cgroup."""
    try:
        with open("/proc/self/cgroup") as f:
            rel = [l.strip().split(":", 2)[2] for l in f if l.startswith("0::")][0]
    except (OSError, IndexError):
        return None, None
    path = "/sys/fs/cgroup" + rel
    probe = path
    while probe.startswith("/sys/fs/cgroup/"):
        if os.path.basename(probe).startswith("job_"):
            return probe, "Slurm job cgroup"
        probe = os.path.dirname(probe)
    return path, "process cgroup (not inside a Slurm job)"


def _read_int(path):
    try:
        with open(path) as f:
            v = f.read().strip()
        return None if v == "max" else int(v)
    except (OSError, ValueError):
        return None


def gpu_memory_stats():
    """XLA allocator stats of the first JAX device: in use, peak, pool limit (bytes).
    None when the device does not report them (e.g. running on CPU)."""
    import jax
    stats = jax.devices()[0].memory_stats() or {}
    out = {k: stats.get(k) for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")}
    return out if any(v is not None for v in out.values()) else None


def basic_memory_summary():
    """One-shot memory summary from peaks already tracked by the kernel and XLA
    (no sampling, so no runtime cost). Used when detailed monitoring is off."""
    main_peak = _read_kb("/proc/self/status", "VmHWM:")
    cgroup, kind = _job_cgroup()
    cg_peak = _read_int(f"{cgroup}/memory.peak") if cgroup else None
    try:
        gpu = gpu_memory_stats()
    except Exception:
        gpu = None
    lines = ["Memory usage summary:",
             f"  main process peak RSS        : {main_peak/GB:.2f} GB"]
    if cg_peak is not None:
        lines.append(f"  total job peak ({kind}): {cg_peak/GB:.2f} GB"
                     " (all processes incl. GRIB2 writers, plus file page cache)")
    if gpu and gpu.get("peak_bytes_in_use") is not None:
        lines.append(f"  GPU peak in use (XLA)        : {gpu['peak_bytes_in_use']/GB:.2f} GB")
    print("\n".join(lines), flush=True)


class GpuStepTimer:
    """Measures the device compute time of each rollout step without stalling the
    pipeline: a watcher thread blocks on each step's outputs as soon as the step is
    queued and timestamps when they become ready. Steps run back to back on the GPU,
    so compute(i) = ready(i) - max(queued(i), ready(i-1))."""

    def __init__(self):
        self.queue = queue.Queue()
        self.events = {}
        self.compute = {}
        self.thread = threading.Thread(target=self._run, name="gpu-step-timer", daemon=True)
        self.thread.start()

    def launched(self, key, outputs):
        """Call right after the step producing `outputs` (a JAX pytree) was queued."""
        event = threading.Event()
        self.events[key] = event
        self.queue.put((key, event, time(), outputs))

    def _run(self):
        import jax
        prev_ready = 0.0
        while True:
            item = self.queue.get()
            if item is None:
                return
            key, event, t_queued, outputs = item
            try:
                for leaf in jax.tree_util.tree_leaves(outputs):
                    if hasattr(leaf, "block_until_ready"):
                        leaf.block_until_ready()
                t_ready = time()
                self.compute[key] = t_ready - max(t_queued, prev_ready)
                prev_ready = t_ready
            except Exception:
                self.compute[key] = None
            finally:
                del outputs, item
                event.set()

    def compute_time(self, key):
        """Device compute seconds of step `key` (waits until it is known), or None."""
        self.events.pop(key).wait()
        return self.compute.pop(key)

    def close(self):
        self.queue.put(None)


def nvidia_smi_used(pid):
    """GPU memory (bytes) nvidia-smi attributes to pid (XLA pool + CUDA context), or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        fields = [x.strip() for x in line.split(",")]
        if len(fields) == 2 and fields[0] == str(pid):
            return int(fields[1]) * 1024 ** 2
    return None


class ResourceMonitor:
    """Samples host memory every `interval` seconds in a daemon thread and keeps the
    peak per phase. Call phase(name) to start a new phase, report() at the end."""

    def __init__(self, first_phase="startup", interval=2.0):
        self.interval = interval
        self.pid = os.getpid()
        self.cgroup, self.cgroup_kind = _job_cgroup()
        self.lock = threading.Lock()
        self.current = first_phase
        self.order = [first_phase]
        self.peaks = {first_phase: {"main": 0, "workers": 0, "total": 0, "cgroup": 0}}
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self.thread.start()

    def phase(self, name):
        self.sample()
        with self.lock:
            self.current = name
            self.order.append(name)
            self.peaks[name] = {"main": 0, "workers": 0, "total": 0, "cgroup": 0}

    def sample(self):
        main = _read_kb(f"/proc/{self.pid}/status", "VmRSS:")
        workers = sum(_pss(p) for p in _descendants(self.pid))
        cg = _read_int(f"{self.cgroup}/memory.current") if self.cgroup else None
        with self.lock:
            peak = self.peaks[self.current]
            peak["main"] = max(peak["main"], main)
            peak["workers"] = max(peak["workers"], workers)
            peak["total"] = max(peak["total"], _pss(self.pid) + workers)
            peak["cgroup"] = max(peak["cgroup"], cg or 0)

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def report(self, gpu_stats=None, gpu_smi=None):
        self.sample()
        self.stop_event.set()
        lines = ["=" * 78, "Memory usage report", "-" * 78,
                 "Host (CPU) memory peak per phase, GB:",
                 f"  {'phase':<26}{'main RSS':>10}{'writers PSS':>13}{'tree PSS':>11}{'cgroup':>10}"]
        overall = {"main": 0, "workers": 0, "total": 0, "cgroup": 0}
        for name in self.order:
            p = self.peaks[name]
            lines.append(f"  {name:<26}{p['main']/GB:>10.2f}{p['workers']/GB:>13.2f}"
                         f"{p['total']/GB:>11.2f}{p['cgroup']/GB:>10.2f}")
            for k in overall:
                overall[k] = max(overall[k], p[k])
        lines.append("  (main RSS: python/JAX process; writers PSS: fork server + GRIB2 workers;")
        lines.append("   tree PSS: whole process tree without double counting shared pages;")
        lines.append("   cgroup: memory charged to the job, including file page cache)")

        if gpu_stats:
            lines += ["-" * 78, "GPU memory (device 0), GB:"]
            if gpu_stats.get("peak_bytes_in_use") is not None:
                lines.append(f"  XLA peak in use        : {gpu_stats['peak_bytes_in_use']/GB:.2f}"
                             "   <- what the model actually needs")
            if gpu_stats.get("bytes_in_use") is not None:
                lines.append(f"  XLA in use now         : {gpu_stats['bytes_in_use']/GB:.2f}")
            if gpu_stats.get("bytes_limit") is not None:
                lines.append(f"  XLA pool size          : {gpu_stats['bytes_limit']/GB:.2f}"
                             "   (preallocated, XLA_PYTHON_CLIENT_MEM_FRACTION)")
            if gpu_smi is not None:
                lines.append(f"  nvidia-smi for process : {gpu_smi/GB:.2f}   (pool + CUDA context)")

        cg_peak = _read_int(f"{self.cgroup}/memory.peak") if self.cgroup else None
        lines += ["-" * 78, "Total host memory for the job:"]
        lines.append(f"  process tree peak (PSS): {overall['total']/GB:.2f} GB")
        if cg_peak is not None:
            lines.append(f"  {self.cgroup_kind} memory.peak: {cg_peak/GB:.2f} GB"
                         " (includes reclaimable page cache)")
        lines.append(f"  suggested #SBATCH --mem : {int(overall['total'] * 1.25 / GB) + 1}G"
                     " (tree peak + 25% margin)")
        lines.append("=" * 78)
        print("\n".join(lines), flush=True)
