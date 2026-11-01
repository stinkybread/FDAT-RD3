#!/usr/bin/env python3
"""TensorRT 11 per-layer profiler for VapourKit-built engines (trtexec --dumpProfile equivalent).

VapourKit ships TensorRT as pip wheels (no trtexec.exe); its data/build_trt_engine.py turns
trtexec-style args into a strongly typed TRT 11 build (ModelOpt BF16/FP16 conversion + FP32
fallback islands). This script imports that exact builder, so the engine profiled here is
built the way vs-mlrt builds your deployment engines -- the only change is
ProfilingVerbosity.DETAILED, which keeps ONNX layer names in the engine.

Run with VapourKit's Python:

    <VapourKit>\\data\\vapoursynth-portable\\python.exe profile_trt.py --vk-data <VapourKit>\\data ^
        --onnx "C:\\path\\model_static_540x720.onnx" --bf16

Outputs (folder <onnx stem>_profile next to the ONNX, or --out):
    summary.txt        what is printed to the console
    layers.csv         every engine layer: median ms, %, type, component, fused ONNX ops
    profile.json       raw per-iteration layer times + latency samples
    engine_info.json   TensorRT engine inspector dump (tactics, formats, fusions)

Default build args mirror vsmlrt.Backend.TRT defaults: builder optimization level 3, no TF32,
fp32 CHW input/output. Pass --fp16 or --bf16 to match how you deploy. --engine profiles an
existing engine instead of building (layer names will be opaque unless it was built with
detailed profiling verbosity).
"""

import argparse
import csv
import os
import importlib.util
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# VapourKit data folder (holds build_trt_engine.py). Set VAPOURKIT_DATA or pass --vk-data.
DEFAULT_VK_DATA = os.environ.get("VAPOURKIT_DATA")

# ----------------------------------------------------------------------------------------------
# Attribution: engine layer -> model component.  Pure functions (no TensorRT needed).
# ----------------------------------------------------------------------------------------------

LAYOUT_OPS = ("Transpose", "Reshape", "Flatten", "Squeeze", "Unsqueeze", "Gather", "Slice",
              "Concat", "Split", "Expand", "Pad", "DepthToSpace", "SpaceToDepth", "Shape",
              "Identity", "Cast", "Tile")
ONNX_RE = re.compile(r"ONNX Layer:\s*([^\]\x1e]+)")
FOREIGN_RE = re.compile(r"ForeignNode\[(.*)\]")
BLOCK_RE = re.compile(r"/groups(?:/groups)?\.(\d+)/blocks\.(\d+)/([^/]+)")
GROUPCONV_RE = re.compile(r"/groups(?:/groups)?\.(\d+)/conv/")
SUB = {"attn": "attn", "conv": "dwconv", "inter": "aim_gate", "ffn": "ffn", "n1": "norm", "n2": "norm"}


def onnx_op_type(node_name):
    leaf = node_name.rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"_\d+$", "", leaf)


def component_of(node_name, pattern):
    """'/groups/groups.0/blocks.3/attn/MatMul' -> 'spatial.attn' (pattern-dependent)."""
    n = node_name
    m = BLOCK_RE.search(n)
    if m:
        btype = pattern[int(m.group(2)) % len(pattern)]
        return f"{btype}.{SUB.get(m.group(3), 'glue')}"
    if GROUPCONV_RE.search(n):
        return "group_conv"
    for key, comp in (("/pre/", "pre_resize"), ("/conv_first", "stem"), ("/conv_after", "conv_after"),
                      ("/upsampler", "upsampler")):
        if key in n:
            return comp
    if n.startswith("/groups"):
        return "group_glue"
    return "top_level"  # input pad / crop / global residual add / IO


def kind_of(onnx_nodes, layer_type, layer_name):
    lt = (layer_type or "").lower()
    if "reformat" in lt or layer_name.startswith("Reformatting CopyNode"):
        return "reformat"
    if not onnx_nodes:
        return "shuffle" if "shuffle" in lt else "other"
    if all(onnx_op_type(x).startswith(LAYOUT_OPS) for x in onnx_nodes):
        return "layout"
    return "compute"


def onnx_nodes_of(layer_name, metadata):
    nodes = ONNX_RE.findall(metadata or "")
    if not nodes:
        m = FOREIGN_RE.search(layer_name)
        if m:
            nodes = [s for s in re.split(r"\.\.\.|\+", m.group(1)) if s.strip().startswith("/")]
    if not nodes:
        m = re.search(r"(/[\w./\-]+)", layer_name)
        if m:
            nodes = [m.group(1)]
    return [s.strip() for s in nodes]


def attribute(layer_name, layer_type, metadata, pattern):
    nodes = onnx_nodes_of(layer_name, metadata)
    comps = Counter(component_of(n, pattern) for n in nodes) if nodes else Counter({"unknown": 1})
    if layer_name.startswith("Reformatting CopyNode"):
        m = re.search(r"to (/[\w./\-]+)", layer_name)
        if m:
            comps = Counter({component_of(m.group(1), pattern): 1})
    top, cnt = comps.most_common(1)[0]
    total = sum(comps.values())
    comp = top if cnt == total else f"mixed[{top} {cnt}/{total}]"
    return comp, top, kind_of(nodes, layer_type, layer_name), nodes


# ----------------------------------------------------------------------------------------------
# Build (VapourKit's builder, detailed verbosity)
# ----------------------------------------------------------------------------------------------

def load_vk_builder(vk_data):
    path = Path(vk_data) / "build_trt_engine.py"
    if not path.exists():
        sys.exit(f"VapourKit builder not found: {path}  (use --vk-data)")
    spec = importlib.util.spec_from_file_location("vk_build_trt_engine", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    orig = mod._configure_builder

    def configure_detailed(builder, network, opts, trt, *a, **k):
        config, cache = orig(builder, network, opts, trt, *a, **k)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        return config, cache

    mod._configure_builder = configure_detailed  # looked up at call time by build_from_memory
    return mod


def build_engine(args, engine_path):
    mod = load_vk_builder(args.vk_data)
    # Reuse the FP32-fallback blacklists VapourKit learned for deployment engines (keyed by graph
    # structure), so the profiled engine gets the same precision islands and skips re-probing.
    src_bl = Path(args.vk_data) / "models" / ".vapourkit-trt-blacklists"
    dst_bl = Path(engine_path).parent / ".vapourkit-trt-blacklists"
    if src_bl.is_dir():
        dst_bl.mkdir(exist_ok=True)
        for f in src_bl.iterdir():
            if f.is_file() and not (dst_bl / f.name).exists():
                (dst_bl / f.name).write_bytes(f.read_bytes())
    argv = [f"--onnx={args.onnx}", f"--saveEngine={engine_path}",
            f"--builderOptimizationLevel={args.opt_level}", "--noTF32",
            "--inputIOFormats=fp32:chw", "--outputIOFormats=fp32:chw",
            f"--timingCacheFile={engine_path}.cache", f"--device={args.device}"]
    if args.bf16:
        argv.append("--bf16")
    if args.fp16:
        argv.append("--fp16")
    if args.shapes:
        argv.append(f"--shapes={args.shapes}")
    argv += args.extra
    print("[profile] building with VapourKit builder:", " ".join(argv), flush=True)
    t0 = time.time()
    rc = mod.build(mod.parse_args(argv))
    if rc not in (0, None) or not Path(engine_path).exists():
        sys.exit(f"[profile] engine build failed (rc={rc})")
    print(f"[profile] built in {time.time() - t0:.0f}s -> {engine_path}", flush=True)


# ----------------------------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------------------------

def run(args, engine_path, out_dir):
    import tensorrt as trt
    import torch

    torch.cuda.set_device(args.device)
    dt_map = {}
    for trt_name, torch_dt in (("FLOAT", torch.float32), ("HALF", torch.float16), ("BF16", torch.bfloat16),
                               ("INT32", torch.int32), ("INT64", torch.int64), ("INT8", torch.int8),
                               ("BOOL", torch.bool), ("UINT8", torch.uint8)):
        if hasattr(trt.DataType, trt_name):
            dt_map[getattr(trt.DataType, trt_name)] = torch_dt

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    engine = trt.Runtime(logger).deserialize_cuda_engine(Path(engine_path).read_bytes())
    if engine is None:
        sys.exit("[profile] could not deserialize engine (built by a different TensorRT version?)")
    ctx = engine.create_execution_context()

    shapes = {}
    if args.shapes:
        for part in args.shapes.split(","):
            name, dims = part.rsplit(":", 1)
            shapes[name] = tuple(int(d) for d in dims.split("x"))
    bufs, io_desc = {}, []
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    for name in names:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            shp = tuple(engine.get_tensor_shape(name))
            if -1 in shp:
                if name not in shapes:
                    sys.exit(f"[profile] input '{name}' is dynamic {shp}; pass --shapes {name}:1x3xHxW")
                ctx.set_input_shape(name, shapes[name])
    for name in names:
        is_in = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        shp = tuple(ctx.get_tensor_shape(name))
        dt = dt_map[engine.get_tensor_dtype(name)]
        t = (torch.rand(shp, device="cuda").to(dt) if is_in else torch.empty(shp, dtype=dt, device="cuda"))
        bufs[name] = t
        ctx.set_tensor_address(name, t.data_ptr())
        io_desc.append(f"{'in ' if is_in else 'out'} {name} {list(shp)} {dt}")

    stream = torch.cuda.Stream()
    sh = stream.cuda_stream

    def enqueue():
        if not ctx.execute_async_v3(sh):
            sys.exit("[profile] execute_async_v3 failed")

    # 1) clean end-to-end timing, no profiler attached
    for _ in range(args.warmup):
        enqueue()
    stream.synchronize()
    lat = []
    for _ in range(args.iterations):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(stream)
        enqueue()
        e.record(stream)
        e.synchronize()
        lat.append(s.elapsed_time(e))

    # 2) separate per-layer profiling run
    class Prof(trt.IProfiler):
        def __init__(self):
            trt.IProfiler.__init__(self)
            self.t = defaultdict(list)
            self.order = []

        def report_layer_time(self, layer_name, ms):
            if layer_name not in self.t:
                self.order.append(layer_name)
            self.t[layer_name].append(ms)

    prof = Prof()
    ctx.profiler = prof
    ctx.enqueue_emits_profile = False
    for _ in range(args.profile_iterations):
        enqueue()
        stream.synchronize()
        ctx.report_to_profiler()

    insp = engine.create_engine_inspector()
    insp.execution_context = ctx
    info = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
    meta = {}
    for L in info.get("Layers", []):
        if isinstance(L, dict):
            meta[L.get("Name", "")] = (L.get("LayerType", ""), L.get("Metadata", ""), L.get("TacticName", ""))

    return trt.__version__, io_desc, lat, prof, info, meta


def report(args, out_dir, trt_version, io_desc, lat, prof, info, meta):
    pattern = [p.strip() for p in args.pattern.split(",")]
    rows = []
    for name in prof.order:
        ms = statistics.median(prof.t[name])
        ltype, md, tactic = meta.get(name, ("", "", ""))
        comp, top, kind, nodes = attribute(name, ltype, md, pattern)
        rows.append(dict(layer=name, median_ms=ms, type=ltype, component=comp, top=top, kind=kind,
                         tactic=tactic, n_onnx=len(nodes), onnx_first=nodes[0] if nodes else "",
                         onnx_last=nodes[-1] if nodes else ""))
    layer_sum = sum(r["median_ms"] for r in rows) or 1e-9
    lat_sorted = sorted(lat)
    med = statistics.median(lat)
    p90 = lat_sorted[int(0.9 * (len(lat_sorted) - 1))]

    lines = []
    P = lines.append
    P(f"TensorRT {trt_version} | {Path(args.onnx or args.engine).name} | "
      f"{'bf16' if args.bf16 else 'fp16' if args.fp16 else 'fp32'} | opt level {args.opt_level}")
    for d in io_desc:
        P(f"  {d}")
    P("")
    P(f"End-to-end (no profiler, {len(lat)} iters): median {med:.3f} ms | mean {statistics.mean(lat):.3f} | "
      f"p90 {p90:.3f} | min {lat_sorted[0]:.3f}  ->  {1000 / med:.1f} fps")
    P(f"Per-layer sum (profiled run): {layer_sum:.3f} ms over {len(rows)} engine layers "
      f"(profiling serialises layers, so this exceeds end-to-end)")

    def table(title, key):
        agg = defaultdict(float)
        for r in rows:
            agg[r[key]] += r["median_ms"]
        P("")
        P(title)
        for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
            P(f"  {k:<34s} {v:8.3f} ms  {100 * v / layer_sum:5.1f}%")

    table("By component (mixed fused layers counted under their majority component):", "top")
    table("By kind (layout = pure Transpose/Reshape/Slice..., reformat = TRT copy nodes):", "kind")
    table("By engine layer type:", "type")

    mixed = [r for r in rows if r["component"].startswith("mixed")]
    if mixed:
        mt = sum(r["median_ms"] for r in mixed)
        P("")
        P(f"Fused layers spanning several components: {len(mixed)} layers, {mt:.3f} ms "
          f"({100 * mt / layer_sum:.1f}%) -- attribution there is approximate")

    P("")
    P(f"Top {args.top} layers:")
    for r in sorted(rows, key=lambda r: -r["median_ms"])[: args.top]:
        span = r["onnx_first"] if r["n_onnx"] <= 1 else f"{r['onnx_first']} .. {r['onnx_last']} ({r['n_onnx']} ops)"
        P(f"  {r['median_ms']:7.3f} ms {100 * r['median_ms'] / layer_sum:5.1f}%  {r['type']:<14s} "
          f"{r['component']:<26s} {span[:110]}")

    text = "\n".join(lines)
    print("\n" + text)
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    with open(out_dir / "layers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["pct"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "pct": round(100 * r["median_ms"] / layer_sum, 3)})
    (out_dir / "profile.json").write_text(json.dumps(
        {"latency_ms": lat, "layers": {n: prof.t[n] for n in prof.order}}, indent=1), encoding="utf-8")
    (out_dir / "engine_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"\n[profile] wrote summary.txt, layers.csv, profile.json, engine_info.json -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--onnx", help="ONNX to build (with VapourKit's builder) and profile")
    src.add_argument("--engine", help="profile an existing engine instead of building")
    prec = ap.add_mutually_exclusive_group()
    prec.add_argument("--bf16", action="store_true")
    prec.add_argument("--fp16", action="store_true")
    ap.add_argument("--opt-level", type=int, default=3, help="builder optimization level (vs-mlrt default 3)")
    ap.add_argument("--shapes", help="name:1x3xHxW for dynamic ONNX (static ONNX needs nothing)")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--profile-iterations", type=int, default=50)
    ap.add_argument("--pattern", default="spatial,channel,dictionary",
                    help="per-group block pattern, for naming blocks.N (FDAT/FDAT-RD default)")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--vk-data", default=DEFAULT_VK_DATA,
                    help="VapourKit data folder (default: $VAPOURKIT_DATA)")
    ap.add_argument("--out", help="output folder")
    ap.add_argument("extra", nargs="*", help="extra trtexec-style args passed to the builder (after --)")
    args = ap.parse_args()
    if args.onnx and not args.vk_data:
        ap.error("--vk-data (or VAPOURKIT_DATA) is required to build an engine")

    src_path = Path(args.onnx or args.engine)
    if not src_path.exists():
        sys.exit(f"not found: {src_path}")
    out_dir = Path(args.out) if args.out else src_path.parent / f"{src_path.stem}_profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.onnx:
        prec = "bf16" if args.bf16 else "fp16" if args.fp16 else "fp32"
        engine_path = out_dir / f"{src_path.stem}_{prec}_detailed.engine"
        build_engine(args, str(engine_path))
    else:
        engine_path = src_path
    report(args, out_dir, *run(args, str(engine_path), out_dir))


if __name__ == "__main__":
    main()
