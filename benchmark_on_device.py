"""Device benchmark for a compressed model (Raspberry Pi, Jetson, laptop).

Measures, with one declared protocol: median and 95th percentile latency, peak resident memory,
CPU temperature before and after, and the throttling flag (Raspberry Pi only, through vcgencmd).
Energy needs an external USB power meter: pass the average power read on the meter with --power-w
and the script reports energy per image.

Supported model files:
  .pt / .ts   TorchScript module (torch.jit.load)
  .onnx       ONNX model (needs onnxruntime)

Examples (on the device):
  python benchmark_on_device.py --model student_int8.ts --input 1 3 800 1200 --threads 4 --runs 30
  python benchmark_on_device.py --model student.onnx --input 1 3 800 1200 --threads 4 --runs 30 --power-w 7.2
  python benchmark_on_device.py --selftest          # small self test, no model file needed

The input is random noise of the requested shape. Latency of convolutional networks does not depend on the pixel
values, but detector post-processing (NMS) does depend on the number of candidate boxes, so a final measurement
must also be made with real images (use --image-dir).
"""
import argparse, json, os, platform, statistics, subprocess, sys, time
from pathlib import Path

def read_temp_c():
    for p in ("/sys/class/thermal/thermal_zone0/temp",):
        try:
            return int(Path(p).read_text().strip()) / 1000.0
        except Exception:
            pass
    return None

def read_throttled():
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=3).stdout.strip()
        return out or None
    except Exception:
        return None

def rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:
        try:                                    # fallback without psutil (Linux)
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
        except Exception:
            return None

def load_runner(path, threads):
    if path.endswith((".pt", ".ts")):
        import torch
        torch.set_num_threads(threads)
        m = torch.jit.load(path, map_location="cpu").eval()
        def run(x):
            with torch.no_grad():
                return m(torch.from_numpy(x))
        return run
    if path.endswith(".onnx"):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        name = s.get_inputs()[0].name
        return lambda x: s.run(None, {name: x})
    raise SystemExit("unsupported model file (use .pt, .ts or .onnx)")

def selftest_model(tmp):
    import torch, torch.nn as nn
    m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 3)).eval()
    p = str(Path(tmp) / "selftest.ts")
    torch.jit.trace(m, torch.zeros(1, 3, 64, 64)).save(p)
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--input", nargs=4, type=int, default=[1, 3, 800, 1200])
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3); ap.add_argument("--power-w", type=float, default=None)
    ap.add_argument("--out", default="device_benchmark.json"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    import numpy as np
    if a.selftest:
        import tempfile
        a.model = selftest_model(tempfile.mkdtemp()); a.input = [1, 3, 64, 64]; a.runs = 10
    if not a.model:
        raise SystemExit("give --model or --selftest")
    run = load_runner(a.model, a.threads)
    x = np.random.RandomState(0).rand(*a.input).astype("float32")
    t_before, thr_before, rss0 = read_temp_c(), read_throttled(), rss_mb()
    for _ in range(a.warmup):
        run(x)
    ts, peak = [], rss0 or 0.0
    for _ in range(a.runs):
        t0 = time.perf_counter(); run(x); ts.append((time.perf_counter() - t0) * 1000.0)
        r = rss_mb(); peak = max(peak, r or 0.0)
    ts_sorted = sorted(ts)
    res = {
        "model": a.model, "input": a.input, "threads": a.threads, "runs": a.runs, "warmup": a.warmup,
        "latency_ms": {"median": statistics.median(ts), "p95": ts_sorted[int(0.95 * (len(ts_sorted) - 1))],
                       "mean": statistics.mean(ts), "std": statistics.pstdev(ts), "min": ts_sorted[0], "max": ts_sorted[-1]},
        "peak_rss_mb": peak, "temp_c_before": t_before, "temp_c_after": read_temp_c(),
        "throttled_before": thr_before, "throttled_after": read_throttled(),
        "platform": {"machine": platform.machine(), "system": platform.platform(), "python": sys.version.split()[0]},
    }
    if a.power_w is not None:
        res["power_w_external_meter"] = a.power_w
        res["energy_j_per_image"] = a.power_w * res["latency_ms"]["median"] / 1000.0
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
