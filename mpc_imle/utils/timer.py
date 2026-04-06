import torch
import os, csv, time, statistics as stats
from collections import defaultdict
import time
import jax
import jax.numpy as jnp

class Timer:

	def __init__(self):
		self._start = time.time()

	def __call__(self, reset=True):
		now = time.time()
		diff = now - self._start
		if reset:
			self._start = now
		return diff
	
def sync_device_from_module(module):
    try:
        dev = next(module.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

# ---------------- Timer_Better ---------------- #

class Timer_Better:
    def __init__(self, device=None, name="section", autolog=True, echo=False):
        self.name = name
        self.autolog = autolog
        self.echo = echo

        # normalize device
        if device is None:
            self.device = None
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self._use_cuda = (self.device is not None and getattr(self.device, "type", None) == "cuda")

        if self._use_cuda:
            torch.cuda.synchronize()
            self._starter = torch.cuda.Event(enable_timing=True)
            self._ender   = torch.cuda.Event(enable_timing=True)
            self._starter.record()
        else:
            self._t0 = time.perf_counter()

    def reset(self):
        if self._use_cuda:
            torch.cuda.synchronize()
            self._starter = torch.cuda.Event(enable_timing=True)
            self._ender   = torch.cuda.Event(enable_timing=True)
            self._starter.record()
        else:
            self._t0 = time.perf_counter()

    def __call__(self, reset=True):
        # returns seconds
        if self._use_cuda:
            self._ender.record()
            torch.cuda.synchronize()
            sec = self._starter.elapsed_time(self._ender) / 1e3
        else:
            sec = time.perf_counter() - self._t0

        if self.autolog:
            TIMER_LOGGER.log(self.name, sec)
        if self.echo:
            print(f"[timer] {self.name}: {sec*1e3:.2f} ms")
        if reset:
            self.reset()
        return sec

def _block_until_ready(x):
    # Works for a single array or a pytree of arrays
    return jax.tree_util.tree_map(lambda a: a.block_until_ready(), x)

class Timer_Better_Jax:
    """
    JAX analogue of Timer_Better.

    Key difference vs Torch:
      - Torch can synchronize a CUDA stream globally.
      - JAX is best synchronized by blocking on a produced device value.

    Usage patterns:
      t = Timer_Better_Jax(name="rank", echo=True)
      y = f(...)             # JAX work
      t(out=y)               # blocks on y, logs elapsed

    If you don't pass `out`, it will fall back to timing host time only
    (not recommended for GPU timing).
    """
    def __init__(self, device=None, name="section", autolog=True, echo=False):
        self.name = name
        self.autolog = autolog
        self.echo = echo

        # Keep this for API parity; JAX device selection is handled elsewhere
        self.device = device

        # Make sure any previously queued work doesn't pollute the start time.
        # We force a tiny device op and block on it.
        _block_until_ready(jnp.array(0, dtype=jnp.int32))

        self._t0 = time.perf_counter()

    def reset(self):
        _block_until_ready(jnp.array(0, dtype=jnp.int32))
        self._t0 = time.perf_counter()

    def __call__(self, out=None, reset=True):
        # If out is provided, block on it to measure "true" device time.
        if out is not None:
            _block_until_ready(out)

        sec = time.perf_counter() - self._t0

        if self.autolog:
            # your TimerLogger uses .log(name, seconds)
            TIMER_LOGGER.log(self.name, sec)

        if self.echo:
            print(f"[timer] {self.name}: {sec*1e3:.2f} ms")

        if reset:
            self.reset()

        return sec


# ---------------- TimerLogger (per-run scoping) ---------------- #

class TimerLogger:
    def __init__(self):
        # current-run samples: metric name -> [seconds, ...]
        self.records = defaultdict(list)
        # current run metadata
        self.run_id = None
        self.run_meta = {}
        # optional accumulated per-run summaries
        self.run_summaries = []
        
    def clear(self):
        self.records.clear()

    # ---- run lifecycle ---- #
    def start_run(self, run_id=None, **meta):
        """Begin a new run; clears sample buffer for clean per-run stats."""
        self.records.clear()
        self.run_id = run_id or f"run_{int(time.time())}"
        self.run_meta = dict(meta)

    def end_run(self, csv_path=None):
        """
        Aggregate current run into a dict of totals/means/etc.
        Optionally append a single row per run to CSV (wide format).
        """
        summary = {"run_id": self.run_id, **self.run_meta}

        # aggregate each metric present in this run
        for name, vals in self.records.items():
            if not vals:
                continue
            total = sum(vals)
            n = len(vals)
            mean = total / n
            std  = stats.pstdev(vals) if n > 1 else 0.0
            p50  = stats.median(vals)
            srt  = sorted(vals)
            p90  = srt[int(0.90*(n-1))] if n > 1 else vals[0]

            # store with clear suffixes in *seconds*
            summary[f"{name}_total_s"] = total
            summary[f"{name}_count"]   = n
            summary[f"{name}_mean_s"]  = mean
            summary[f"{name}_std_s"]   = std
            summary[f"{name}_p50_s"]   = p50
            summary[f"{name}_p90_s"]   = p90

        self.run_summaries.append(summary)

        if csv_path:
            os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
            # stabilize header across runs
            base = ["run_id"]
            others = sorted([k for k in summary.keys() if k not in base])
            fieldnames = base + others
            exists = os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if not exists:
                    w.writeheader()
                # ensure all columns
                row = {k: summary.get(k, "") for k in fieldnames}
                w.writerow(row)

        # keep current samples (optional); often you want a fresh buffer:
        self.records.clear()
        return summary

    # ---- sample-level logging (for current run) ---- #
    def log(self, name, seconds: float):
        self.records[name].append(float(seconds))

    # convenience
    def total(self, name):  return sum(self.records.get(name, ()))
    def count(self, name):  return len(self.records.get(name, ()))
    def mean(self, name):
        vals = self.records.get(name, ())
        return (sum(vals) / len(vals)) if vals else 0.0
    def totals_dict(self):  return {k: sum(v) for k, v in self.records.items()}

    def summary_text(self):
        """Pretty summary for the *current run buffer* (not historical)."""
        lines = []
        for name, vals in sorted(self.records.items()):
            n = len(vals)
            if n == 0:
                continue
            total = sum(vals)
            mean  = stats.mean(vals)
            std   = stats.pstdev(vals) if n > 1 else 0.0
            p50   = stats.median(vals)
            srt   = sorted(vals)
            p90   = srt[int(0.90*(n-1))] if n > 1 else vals[0]
            lines.append(
                f"{name}: total={total*1e3:.2f} ms | mean={mean*1e3:.2f}±{std*1e3:.2f} "
                f"(N={n}, p50={p50*1e3:.2f} ms, p90={p90*1e3:.2f} ms)"
            )
        return "\n".join(lines)

    def flush_to_csv(self, path="logs/timings_samples.csv", extra: dict = None):
        """
        Append *all current-run samples* (one row per sample) to CSV, then clear buffer.
        Use end_run(csv_path=...) if you prefer one row per run.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not self.records:
            return

        rows = []
        for name, vals in self.records.items():
            for v in vals:
                row = {"name": name, "seconds": float(v), "run_id": self.run_id}
                if extra:
                    row.update(extra)
                rows.append(row)

        base = ["run_id", "name", "seconds"]
        others = sorted([k for k in rows[0].keys() if k not in base])
        fieldnames = base + others

        exists = os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            for r in rows:
                for k in fieldnames:
                    r.setdefault(k, "")
                writer.writerow(r)

        self.records.clear()


TIMER_LOGGER = TimerLogger()