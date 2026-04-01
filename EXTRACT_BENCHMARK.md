# Jetson benchmark extraction helper

`extract_benchmark.py` converts `benchmark_ultralytics_jetson.csv` into a smaller, visualization-friendly CSV.

It is designed around these rules:

- only `status == success` rows are aggregated
- repeated runs are collapsed with `mean` or `median` (`mean` by default)
- latency can be taken from `preprocess`, `inference`, `postprocess`, or `end2end`
- settings are keyed by `precision + format_name`
- wide output matches the style of `jetson_detection.csv`
- long output is tidy and easier to plot directly in Python / pandas / seaborn / matplotlib

The script also tolerates the common source typos `percision` -> `precision` and `speed_posprocess_ms` -> `speed_postprocess_ms`.

## Basic usage

```bash
python extract_benchmark.py benchmark_ultralytics_jetson.csv \
  -o jetson_detection_extracted.csv
```

Default behavior:

- `--mode wide`
- `--agg mean`
- `--latency inference`
- `--latency-unit s`
- all models
- all settings

This produces one row per `network + setting`, with columns like:

```text
network,setting,native_resolution,map_0.6x,map_0.8x,map_1x,map_1.2x,inf_0.6x,inf_0.8x,inf_1x,inf_1.2x
```

## Make the output look like `jetson_detection.csv`

Use a single setting and set a custom latency prefix:

```bash
python extract_benchmark.py benchmark_ultralytics_jetson.csv \
  -o jetson_detection_like.csv \
  --settings fp32-pytorch \
  --latency inference \
  --latency-unit s \
  --latency-prefix p20 \
  --date 2026
```

That gives columns in the style:

```text
network,date,native_resolution,map_0.6x,map_0.8x,map_1x,map_1.2x,p20_0.6x,p20_0.8x,p20_1x,p20_1.2x
```

## Filter specific settings

```bash
python extract_benchmark.py benchmark_ultralytics_jetson.csv \
  -o selected_settings.csv \
  --settings fp32-pytorch fp16-onnx
```

Supported selector styles:

```bash
--settings fp32-pytorch
--settings pytorch-fp32
--settings fp16
--settings onnx
--settings all
```

Notes:

- full combinations select exactly one setting
- precision-only selectors select all matching formats
- format-only selectors select all matching precisions
- multiple selectors are combined as a union

## Split every setting into its own CSV

```bash
python extract_benchmark.py benchmark_ultralytics_jetson.csv \
  -o jetson_detection.csv \
  --split-settings
```

This writes:

```text
jetson_detection_fp32-pytorch.csv
jetson_detection_fp32-torchscript.csv
jetson_detection_fp32-onnx.csv
jetson_detection_fp16-pytorch.csv
jetson_detection_fp16-torchscript.csv
jetson_detection_fp16-onnx.csv
```

You can combine this with filters too:

```bash
python extract_benchmark.py benchmark_ultralytics_jetson.csv \
  -o jetson_detection.csv \
  --settings fp16 fp32-onnx \
  --split-settings
```

## Choose a latency bucket

Inference latency is the default:

```bash
python extract_benchmark.py benchmark_ultralytics_jetson.csv \
  -o inference.csv \
  --latency inference
```

Other options:

```bash
--latency preprocess
--latency postprocess
--latency end2end
```

`end2end` is computed as:

```text
speed_preprocess_ms + speed_inference_ms + speed_postprocess_ms
```

## Use median instead of mean, and include standard deviation

```bash
python extract_benchmark.py benchmark_ultralytics_jetson.csv \
  -o jetson_with_std.csv \
  --agg median \
  --include-std
```

Wide output adds columns like:

```text
map_std_0.6x, ..., inf_std_1.2x
```

Long output adds columns like:

```text
ap50_95_std, latency_std
```

## Long / tidy output

```bash
python extract_benchmark.py benchmark_ultralytics_jetson.csv \
  -o jetson_long.csv \
  --mode long \
  --latency end2end \
  --include-std
```

Long output columns:

```text
network,setting,precision,format_name,native_resolution,imgsz,scale,ap50_95,ap50_95_std,latency,runs,latency_std,latency_bucket,latency_unit
```

This is usually the easiest format for custom plotting.

## Filter models

```bash
python extract_benchmark.py benchmark_ultralytics_jetson.csv \
  -o yolo11_only.csv \
  --models yolo11n,yolo11x \
  --settings fp32-pytorch
```

## Discover available models and settings first

```bash
python extract_benchmark.py benchmark_ultralytics_jetson.csv --list
```

## Notes on scale labels

The script turns `imgsz` into scale labels relative to `--base-imgsz`.

Default base size:

```bash
--base-imgsz 640
```

So the uploaded benchmark file maps to:

```text
384 -> 0.6x
512 -> 0.8x
640 -> 1x
768 -> 1.2x
```

If you want a different reference resolution:

```bash
python extract_benchmark.py benchmark_ultralytics_jetson.csv \
  -o custom_base.csv \
  --base-imgsz 512
```

## Useful real commands for the shared file

Default all-settings wide export:

```bash
python extract_benchmark.py /mnt/data/benchmark_ultralytics_jetson.csv \
  -o /mnt/data/jetson_detection_extracted.csv
```

Target-like export for one setting:

```bash
python extract_benchmark.py /mnt/data/benchmark_ultralytics_jetson.csv \
  -o /mnt/data/jetson_detection_fp32_pytorch_like.csv \
  --settings fp32-pytorch \
  --latency-prefix p20 \
  --date 2026
```

All settings split into separate files:

```bash
python extract_benchmark.py /mnt/data/benchmark_ultralytics_jetson.csv \
  -o /mnt/data/jetson_detection.csv \
  --split-settings
```
