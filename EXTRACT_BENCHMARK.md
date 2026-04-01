# `extract_benchmark.py`

This script turns the raw Ultralytics benchmark CSV into a smaller visualization-friendly CSV.

## What it does

- aggregates repeated runs using `mean` or `median` (`mean` by default)
- only aggregates rows with `status == success`
- warns when some repeats failed and when a full setting/size combination has zero successful runs
- lets you choose the latency bucket to export:
  - `inference` (default)
  - `preprocess`
  - `postprocess`
  - `end-to-end`
- keeps the accuracy metric as `ap50_95`
- renames `gflops` to `flops`
- can emit:
  - a **wide** CSV similar to `jetson_detection_deprecated.csv`
  - a **long/tidy** CSV for plotting tools
- can split output into separate CSV files per `(precision, format_name)` setting

By default, scale labels are computed relative to `--native-imgsz 640`, so the shared Jetson/RTX files become `0.6x`, `0.8x`, `1x`, and `1.2x`.

## Output shapes

### Wide mode

Produces one row per `model + precision + format_name`, with columns like:

- `map_0.6x`, `map_0.8x`, `map_1x`, `map_1.2x`
- `inference_ms_0.6x`, ... or `end_to_end_ms_*`, etc.
- `flops_0.6x`, `flops_0.8x`, `flops_1x`, `flops_1.2x`

### Long mode

Produces one row per `model + precision + format_name + imgsz`, with columns like:

- `model`
- `precision`
- `format_name`
- `imgsz`
- `scale`
- `scale_label`
- `params`
- `flops`
- `ap50_95`
- `inference_ms` / `preprocess_ms` / `postprocess_ms` / `end_to_end_ms`

## Important commands

### 1) List available models and settings

```bash
python extract_benchmark.py benchmark_jetson.csv --list-models --list-settings
```

### 2) Default extraction (wide, mean, inference latency)

```bash
python extract_benchmark.py benchmark_jetson.csv jetson_detection_wide.csv
```

### 3) Long/tidy output with end-to-end latency and standard deviation

```bash
python extract_benchmark.py benchmark_jetson.csv jetson_detection_long.csv \
  --mode long \
  --latency end-to-end \
  --with-std \
  --include-run-counts
```

### 4) Use median instead of mean

```bash
python extract_benchmark.py benchmark_jetson.csv jetson_detection_median.csv \
  --aggregate median
```

### 5) Extract only selected settings

```bash
python extract_benchmark.py benchmark_jetson.csv jetson_selected.csv \
  --settings fp32-pytorch fp16-onnx
```

The setting parser accepts forms like:

- `fp32-pytorch`
- `pytorch/fp32`
- `fp16-onnx`
- `fp16-tensorrt` (works on the shared RTX CSV)

### 6) Extract only selected models

```bash
python extract_benchmark.py benchmark_jetson.csv jetson_yolo11.csv \
  --models yolo11n yolo11x
```

### 7) Split every setting into its own CSV

```bash
python extract_benchmark.py benchmark_jetson.csv outputs/ \
  --mode wide \
  --split-settings
```

This creates files such as:

- `fp32-pytorch.csv`
- `fp16-onnx.csv`
- `fp32-torchscript.csv`

If you pass a file name instead of a directory, the setting is appended to the stem:

```bash
python extract_benchmark.py benchmark_jetson.csv jetson.csv --split-settings
```

which creates files like:

- `jetson_fp32-pytorch.csv`
- `jetson_fp16-onnx.csv`

### 8) Add a constant date/year column to the output

```bash
python extract_benchmark.py benchmark_jetson.csv jetson_detection_2026.csv \
  --date 2026
```

### 9) Change which size counts as `1x`

```bash
python extract_benchmark.py benchmark_jetson.csv jetson_native_512.csv \
  --native-imgsz 512
```

## Commands I verified against the shared CSV files

These ran successfully during validation:

```bash
python extract_benchmark.py benchmark_jetson.csv jetson_detection_wide.csv
python extract_benchmark.py benchmark_jetson.csv jetson_detection_long.csv --mode long --latency end-to-end --with-std --include-run-counts
python extract_benchmark.py benchmark_jetson.csv jetson_split/ --mode wide --split-settings --models yolo11n yolo11x --settings fp32-pytorch fp16-onnx
python extract_benchmark.py benchmark_rtx.csv rtx_trt.csv --settings fp16-tensorrt --models yolo11n yolo11x
```

## Notes

- `flops` keeps the original numeric values from the `gflops` column; only the column name is changed, exactly as requested.
- `--with-std` adds standard deviation columns for `ap50_95` and the selected latency metric.
- When a setting has partial failures, the script still exports the successful repeats and prints warnings.
- In the shared Jetson CSV, `yolov5xu + fp32 + ONNX + imgsz=640` has no successful runs, so that wide cell is left empty/NaN.