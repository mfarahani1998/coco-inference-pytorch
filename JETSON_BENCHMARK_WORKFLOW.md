# Benchmark workflow

## Main ideas
- `bench_ultralytics.py` has two benchmark modes:
  - `--mode full`: prepare missing portable exports and benchmark them
  - `--mode prepared`: use prepared portable exports and still build TensorRT locally when needed
- TensorRT engines are treated as target-local artifacts.
- Repeats are round-robin across the full experiment plan.
- Resume is enabled by default through the CSV output.

## 1) Prepare portable artifacts on a host machine
Ultralytics:

```bash
python prepare_benchmark_artifacts.py --framework ultralytics --portable-only --device auto
```

Torchvision:

```bash
python prepare_benchmark_artifacts.py --framework torchvision --portable-only --device auto
```

Or both in one shot:

```bash
python prepare_benchmark_artifacts.py --framework all --portable-only --device auto
```

Copy the generated cache folders to the target machine:

- `bench_cache_ultralytics/`
- `bench_cache_torchvision/`

## 2) Benchmark on the target with prepared exports
Ultralytics, portable formats only:

```bash
python bench_ultralytics.py \
  --mode prepared \
  --artifact-root bench_cache_ultralytics \
  --formats pytorch torchscript onnx \
  --device 0 \
  --repeats 5
```

## 3) Build TensorRT locally from cached ONNX and benchmark it
Prepare only the local TensorRT engines:

```bash
python prepare_benchmark_artifacts.py \
  --framework ultralytics \
  --formats trt \
  --artifact-root . \
  --device 0
```

Or let the benchmark script build missing engines as part of the run:

```bash
python bench_ultralytics.py \
  --mode prepared \
  --artifact-root bench_cache_ultralytics \
  --formats trt \
  --device 0 \
  --repeats 5
```

## 4) Full one-shot run on a single machine
If you want one command that prepares missing artifacts and immediately benchmarks them:

```bash
python bench_ultralytics.py \
  --mode full \
  --artifact-root bench_cache_ultralytics \
  --device auto \
  --repeats 5
```

## Notes
- Resume is enabled by default. Use `--no-resume` for a fresh rerun.
- Artifact caches are split by model, image size, precision, and format.
- Accuracy is validated once per artifact by default and reused across repeats.
- TensorRT reuse is guarded by sidecar metadata so target-local engines are preferred.
