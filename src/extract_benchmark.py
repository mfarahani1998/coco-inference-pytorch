#!/usr/bin/env python3
"""Extract visualization-friendly CSVs from Ultralytics Jetson benchmark exports.

The input is expected to look like ``benchmark_ultralytics_jetson.csv`` where each
row is one repeated benchmark run for a given model / precision / format / image size.

The script can emit either:
- a wide CSV similar to ``jetson_detection.csv`` (default), or
- a long/tidy CSV that is often easier to plot directly.

Examples
--------
# Default extraction: success-only rows, mean aggregation, inference latency.
python extract_jetson_benchmark.py benchmark_ultralytics_jetson.csv -o jetson_detection_extracted.csv

# Only fp32 PyTorch and fp16 ONNX settings.
python extract_jetson_benchmark.py benchmark_ultralytics_jetson.csv \
    -o selected_settings.csv \
    --settings fp32-pytorch fp16-onnx

# End-to-end latency with standard deviation columns.
python extract_jetson_benchmark.py benchmark_ultralytics_jetson.csv \
    -o jetson_e2e.csv \
    --latency end2end \
    --include-std

# Split each setting into its own CSV.
python extract_jetson_benchmark.py benchmark_ultralytics_jetson.csv \
    -o jetson_detection.csv \
    --split-settings
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


FORMAT_CANONICAL = {
    "pytorch": "PyTorch",
    "pt": "PyTorch",
    "torchscript": "TorchScript",
    "ts": "TorchScript",
    "jit": "TorchScript",
    "onnx": "ONNX",
}

FORMAT_SLUG = {
    "PyTorch": "pytorch",
    "TorchScript": "torchscript",
    "ONNX": "onnx",
    "TensorRT": "TensorRT",
}

PRECISION_CANONICAL = {
    "fp32": "fp32",
    "32": "fp32",
    "float32": "fp32",
    "single": "fp32",
    "fp16": "fp16",
    "16": "fp16",
    "float16": "fp16",
    "half": "fp16",
}

LATENCY_COLUMN_BY_MODE = {
    "preprocess": "speed_preprocess_ms",
    "inference": "speed_inference_ms",
    "postprocess": "speed_postprocess_ms",
}

DEFAULT_LATENCY_PREFIX = {
    "preprocess": "pre",
    "inference": "inf",
    "postprocess": "post",
    "end2end": "e2e",
}

REQUIRED_COLUMNS = {
    "model",
    "imgsz",
    "repeat",
    "precision",
    "format_name",
    "speed_preprocess_ms",
    "speed_inference_ms",
    "speed_postprocess_ms",
    "ap50_95",
    "status",
}

COLUMN_ALIASES = {
    "percision": "precision",
    "speed_posprocess_ms": "speed_postprocess_ms",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract visualization-friendly CSVs from benchmark_ultralytics_jetson.csv. "
            "By default, only successful runs are used and repeated runs are averaged."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to the benchmark CSV (for example benchmark_ultralytics_jetson.csv).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("jetson_detection_extracted.csv"),
        help=(
            "Output CSV path. When --split-settings is used, this becomes the base name. "
            "Example: jetson_detection.csv -> jetson_detection_fp32-pytorch.csv"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("wide", "long"),
        default="wide",
        help="Output layout. 'wide' matches jetson_detection.csv style; 'long' is tidy for plotting.",
    )
    parser.add_argument(
        "--agg",
        choices=("mean", "median"),
        default="mean",
        help="How to collapse repeated runs for the same model / setting / image size.",
    )
    parser.add_argument(
        "--include-std",
        action="store_true",
        help="Also write standard deviation columns across repeats.",
    )
    parser.add_argument(
        "--latency",
        choices=("preprocess", "inference", "postprocess", "end2end"),
        default="inference",
        help=(
            "Which latency bucket to export. 'end2end' = preprocess + inference + postprocess. "
            "Default: inference."
        ),
    )
    parser.add_argument(
        "--latency-unit",
        choices=("ms", "s"),
        default="s",
        help="Latency unit in the exported CSV. Default: s (to match jetson_detection.csv style).",
    )
    parser.add_argument(
        "--latency-prefix",
        default=None,
        help=(
            "Prefix for wide-format latency columns. Defaults to pre / inf / post / e2e. "
            "Example: --latency-prefix p20 to get p20_0.6x columns."
        ),
    )
    parser.add_argument(
        "--settings",
        nargs="*",
        default=None,
        help=(
            "Settings to keep. Accepts full combinations (fp32-pytorch), precision-only selectors "
            "(fp16), format-only selectors (onnx), or 'all'. Can be repeated or comma-separated."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Only keep these models. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--split-settings",
        action="store_true",
        help="Write one CSV per selected setting (for example *_fp32-pytorch.csv).",
    )
    parser.add_argument(
        "--base-imgsz",
        type=int,
        default=640,
        help="Image size treated as 1x when generating scale labels like 0.6x / 0.8x / 1x / 1.2x.",
    )
    parser.add_argument(
        "--native-resolution-label",
        default="1x",
        help="Value written into the native_resolution column in wide mode. Default: 1x.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Optional constant value for a date column in the output (for example 2026).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models, settings, and image sizes from the input CSV, then exit.",
    )
    return parser


def normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for old, new in COLUMN_ALIASES.items():
        if old in df.columns and new not in df.columns:
            renamed[old] = new
    if renamed:
        df = df.rename(columns=renamed)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError("Input CSV is missing required columns: " + ", ".join(missing))


def split_cli_values(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    parts: list[str] = []
    for value in values:
        for piece in re.split(r",", value):
            piece = piece.strip()
            if piece:
                parts.append(piece)
    return parts


def canonicalize_precision(token: str) -> str | None:
    return PRECISION_CANONICAL.get(token.strip().lower())


def canonicalize_format(token: str) -> str | None:
    canonical = FORMAT_CANONICAL.get(token.strip().lower())
    if canonical is None:
        return None
    return FORMAT_SLUG[canonical]


def setting_label_from_row(df: pd.DataFrame) -> pd.Series:
    format_slug = df["format_name"].map(FORMAT_SLUG)
    if format_slug.isna().any():
        bad = sorted(
            df.loc[format_slug.isna(), "format_name"].astype(str).unique().tolist()
        )
        raise ValueError("Unsupported format_name value(s) in CSV: " + ", ".join(bad))
    return df["precision"].astype(str).str.lower() + "-" + format_slug


def preserve_available_order(
    values: Iterable[str], allowed_order: Sequence[str]
) -> list[str]:
    allowed_set = set(values)
    return [item for item in allowed_order if item in allowed_set]


def resolve_setting_selectors(
    selectors: Sequence[str] | None,
    available_settings: Sequence[str],
) -> list[str]:
    raw = split_cli_values(selectors)
    if not raw:
        return list(available_settings)

    raw_norm = [item.strip().lower() for item in raw]
    if any(item == "all" for item in raw_norm):
        return list(available_settings)

    matched: set[str] = set()
    for selector in raw_norm:
        parts = [part for part in re.split(r"[-_:/|+]+", selector) if part]
        if not parts:
            continue

        precision: str | None = None
        fmt_slug: str | None = None

        if len(parts) == 1:
            precision = canonicalize_precision(parts[0])
            fmt_slug = canonicalize_format(parts[0])
            if precision is None and fmt_slug is None:
                raise ValueError(
                    f"Unknown setting selector '{selector}'. Use values like fp32-pytorch, fp16, onnx, or all."
                )
        elif len(parts) == 2:
            first_precision = canonicalize_precision(parts[0])
            second_precision = canonicalize_precision(parts[1])
            first_fmt = canonicalize_format(parts[0])
            second_fmt = canonicalize_format(parts[1])

            if first_precision and second_fmt:
                precision = first_precision
                fmt_slug = second_fmt
            elif second_precision and first_fmt:
                precision = second_precision
                fmt_slug = first_fmt
            else:
                raise ValueError(
                    f"Could not parse selector '{selector}'. Use fp32-pytorch, pytorch-fp32, fp16, or onnx."
                )
        else:
            raise ValueError(
                f"Could not parse selector '{selector}'. Use fp32-pytorch, fp16, onnx, or all."
            )

        for setting in available_settings:
            setting_precision, setting_fmt = setting.split("-", 1)
            if precision is not None and setting_precision != precision:
                continue
            if fmt_slug is not None and setting_fmt != fmt_slug:
                continue
            matched.add(setting)

    if not matched:
        raise ValueError(
            "No settings matched your selectors. Available settings: "
            + ", ".join(available_settings)
        )
    return preserve_available_order(matched, available_settings)


def resolve_model_selectors(
    selectors: Sequence[str] | None,
    available_models: Sequence[str],
) -> list[str]:
    raw = split_cli_values(selectors)
    if not raw:
        return list(available_models)

    requested = [item.strip() for item in raw]
    available_lookup = {model.lower(): model for model in available_models}
    resolved: list[str] = []
    unknown: list[str] = []
    for model in requested:
        key = model.lower()
        canonical = available_lookup.get(key)
        if canonical is None:
            unknown.append(model)
        elif canonical not in resolved:
            resolved.append(canonical)

    if unknown:
        raise ValueError(
            "Unknown model selector(s): "
            + ", ".join(unknown)
            + ". Available models: "
            + ", ".join(available_models)
        )
    return resolved


def choose_latency_series(df: pd.DataFrame, latency_mode: str) -> pd.Series:
    if latency_mode == "end2end":
        return (
            df["speed_preprocess_ms"].astype(float)
            + df["speed_inference_ms"].astype(float)
            + df["speed_postprocess_ms"].astype(float)
        )
    column = LATENCY_COLUMN_BY_MODE[latency_mode]
    return df[column].astype(float)


def format_scale_label(imgsz: float | int, base_imgsz: int) -> str:
    ratio = round(float(imgsz) / float(base_imgsz), 6)
    if abs(ratio - round(ratio)) < 1e-9:
        text = str(int(round(ratio)))
    else:
        text = f"{ratio:.6f}".rstrip("0").rstrip(".")
    return f"{text}x"


def print_available_choices(df: pd.DataFrame) -> None:
    models = df["model"].drop_duplicates().astype(str).tolist()
    settings = df["setting"].drop_duplicates().astype(str).tolist()
    imgsz_values = sorted(df["imgsz"].dropna().astype(int).unique().tolist())

    print("Available models:")
    for model in models:
        print(f"  - {model}")

    print("\nAvailable settings:")
    for setting in settings:
        print(f"  - {setting}")

    print("\nAvailable image sizes:")
    print("  - " + ", ".join(str(value) for value in imgsz_values))


def load_and_prepare_input(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise ValueError(f"Input CSV does not exist: {input_csv}")

    df = pd.read_csv(input_csv)
    df = normalize_input_columns(df)
    validate_columns(df)

    df = df.copy()
    df["precision"] = df["precision"].astype(str).str.lower()
    df["setting"] = setting_label_from_row(df)
    return df


def aggregate_data(
    df: pd.DataFrame,
    agg: str,
    latency_mode: str,
    latency_unit: str,
    include_std: bool,
    base_imgsz: int,
    date_value: str | None,
    native_resolution_label: str,
) -> pd.DataFrame:
    work = df.copy()
    work = work.loc[work["status"].astype(str).str.lower() == "success"].copy()
    if work.empty:
        raise ValueError(
            "No successful rows remained after filtering status == 'success'."
        )

    work["latency_ms_selected"] = choose_latency_series(work, latency_mode)
    work["scale"] = work["imgsz"].apply(
        lambda value: format_scale_label(value, base_imgsz)
    )
    work["network"] = work["model"].astype(str)

    model_order = work["network"].drop_duplicates().tolist()
    setting_order = work["setting"].drop_duplicates().tolist()

    work["network"] = pd.Categorical(
        work["network"], categories=model_order, ordered=True
    )
    work["setting"] = pd.Categorical(
        work["setting"], categories=setting_order, ordered=True
    )

    group_cols = [
        "network",
        "setting",
        "precision",
        "format_name",
        "imgsz",
        "scale",
    ]

    group = work.groupby(group_cols, sort=False, observed=True)

    if agg == "mean":
        ap_series = group["ap50_95"].mean()
        latency_series = group["latency_ms_selected"].mean()
    elif agg == "median":
        ap_series = group["ap50_95"].median()
        latency_series = group["latency_ms_selected"].median()
    else:
        raise ValueError(f"Unsupported aggregation mode: {agg}")

    result = pd.concat(
        [
            ap_series.rename("ap50_95"),
            latency_series.rename("latency_ms"),
            group.size().rename("runs"),
        ],
        axis=1,
    ).reset_index()

    if include_std:
        std_df = pd.concat(
            [
                group["ap50_95"].std(ddof=1).rename("ap50_95_std"),
                group["latency_ms_selected"].std(ddof=1).rename("latency_std_ms"),
            ],
            axis=1,
        ).reset_index()
        result = result.merge(std_df, on=group_cols, how="left")

    if latency_unit == "s":
        result["latency"] = result["latency_ms"] / 1000.0
        if include_std:
            result["latency_std"] = result["latency_std_ms"] / 1000.0
    elif latency_unit == "ms":
        result["latency"] = result["latency_ms"]
        if include_std:
            result["latency_std"] = result["latency_std_ms"]
    else:
        raise ValueError(f"Unsupported latency unit: {latency_unit}")

    result["native_resolution"] = native_resolution_label
    if date_value is not None:
        result["date"] = date_value

    return result


def build_long_output(
    aggregated: pd.DataFrame,
    latency_mode: str,
    latency_unit: str,
    include_std: bool,
) -> pd.DataFrame:
    columns = ["network", "setting", "precision", "format_name"]
    if "date" in aggregated.columns:
        columns.append("date")
    columns.extend(
        [
            "native_resolution",
            "imgsz",
            "scale",
            "ap50_95",
        ]
    )
    if include_std:
        columns.append("ap50_95_std")
    columns.extend(["latency", "runs"])
    if include_std:
        columns.append("latency_std")
    columns.extend(["latency_bucket", "latency_unit"])

    long_df = aggregated.copy()
    long_df["latency_bucket"] = latency_mode
    long_df["latency_unit"] = latency_unit
    long_df = long_df[columns]
    long_df = long_df.sort_values(
        ["network", "setting", "imgsz"], kind="stable"
    ).reset_index(drop=True)
    return long_df


def _pivot_metric(
    aggregated: pd.DataFrame,
    index_cols: list[str],
    value_column: str,
    output_prefix: str,
    scale_order: Sequence[str],
) -> pd.DataFrame:
    pivot = aggregated.pivot(index=index_cols, columns="scale", values=value_column)
    pivot = pivot.reindex(columns=list(scale_order))
    pivot.columns = [f"{output_prefix}_{scale}" for scale in pivot.columns]
    return pivot


def build_wide_output(
    aggregated: pd.DataFrame,
    include_std: bool,
    latency_prefix: str,
) -> pd.DataFrame:
    scale_order = (
        aggregated[["imgsz", "scale"]]
        .drop_duplicates()
        .sort_values("imgsz", kind="stable")["scale"]
        .tolist()
    )

    multiple_settings = aggregated["setting"].nunique(dropna=False) > 1
    index_cols = ["network"]
    if multiple_settings:
        index_cols.append("setting")
    if "date" in aggregated.columns:
        index_cols.append("date")
    index_cols.append("native_resolution")

    pieces = [
        _pivot_metric(aggregated, index_cols, "ap50_95", "map", scale_order),
        _pivot_metric(aggregated, index_cols, "latency", latency_prefix, scale_order),
    ]
    if include_std:
        pieces.extend(
            [
                _pivot_metric(
                    aggregated, index_cols, "ap50_95_std", "map_std", scale_order
                ),
                _pivot_metric(
                    aggregated,
                    index_cols,
                    "latency_std",
                    f"{latency_prefix}_std",
                    scale_order,
                ),
            ]
        )

    wide_df = pd.concat(pieces, axis=1).reset_index()

    ordered_cols = list(index_cols)
    ordered_cols.extend([f"map_{scale}" for scale in scale_order])
    ordered_cols.extend([f"{latency_prefix}_{scale}" for scale in scale_order])
    if include_std:
        ordered_cols.extend([f"map_std_{scale}" for scale in scale_order])
        ordered_cols.extend([f"{latency_prefix}_std_{scale}" for scale in scale_order])

    wide_df = wide_df[ordered_cols]
    wide_df = wide_df.sort_values(index_cols, kind="stable").reset_index(drop=True)
    return wide_df


def ensure_parent_dir(path: Path) -> None:
    parent = path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def base_output_path(path: Path) -> tuple[Path, str, str]:
    directory = path.parent if str(path.parent) not in ("", ".") else Path(".")
    if path.suffix:
        stem = path.stem
        suffix = path.suffix
    else:
        stem = path.name
        suffix = ".csv"
    return directory, stem, suffix


def write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_parent_dir(path)
    df.to_csv(path, index=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        source_df = load_and_prepare_input(args.input_csv)

        if args.list:
            print_available_choices(source_df)
            return 0

        available_settings = source_df["setting"].drop_duplicates().tolist()
        available_models = source_df["model"].drop_duplicates().tolist()

        selected_settings = resolve_setting_selectors(args.settings, available_settings)
        selected_models = resolve_model_selectors(args.models, available_models)

        filtered = source_df.loc[
            source_df["setting"].isin(selected_settings)
            & source_df["model"].isin(selected_models)
        ].copy()
        if filtered.empty:
            raise ValueError(
                "No rows remained after applying the requested model / setting filters."
            )

        aggregated = aggregate_data(
            df=filtered,
            agg=args.agg,
            latency_mode=args.latency,
            latency_unit=args.latency_unit,
            include_std=args.include_std,
            base_imgsz=args.base_imgsz,
            date_value=args.date,
            native_resolution_label=args.native_resolution_label,
        )

        latency_prefix = args.latency_prefix or DEFAULT_LATENCY_PREFIX[args.latency]

        if args.split_settings:
            out_dir, stem, suffix = base_output_path(args.output)
            ensure_directory(out_dir)
            written_paths: list[Path] = []

            for setting in selected_settings:
                subset = aggregated.loc[
                    aggregated["setting"].astype(str) == setting
                ].copy()
                if subset.empty:
                    continue
                if args.mode == "wide":
                    export_df = build_wide_output(
                        aggregated=subset,
                        include_std=args.include_std,
                        latency_prefix=latency_prefix,
                    )
                else:
                    export_df = build_long_output(
                        aggregated=subset,
                        latency_mode=args.latency,
                        latency_unit=args.latency_unit,
                        include_std=args.include_std,
                    )
                out_path = out_dir / f"{stem}_{setting}{suffix}"
                write_csv(out_path, export_df)
                written_paths.append(out_path)

            if not written_paths:
                raise ValueError(
                    "No split CSVs were written; all selected settings were empty after filtering."
                )

            print("Wrote files:")
            for path in written_paths:
                print(f"  - {path}")
        else:
            if args.mode == "wide":
                export_df = build_wide_output(
                    aggregated=aggregated,
                    include_std=args.include_std,
                    latency_prefix=latency_prefix,
                )
            else:
                export_df = build_long_output(
                    aggregated=aggregated,
                    latency_mode=args.latency,
                    latency_unit=args.latency_unit,
                    include_std=args.include_std,
                )
            write_csv(args.output, export_df)
            print(f"Wrote {args.output}")

        return 0
    except (
        Exception
    ) as exc:  # pragma: no cover - surfaced through argparse for CLI users.
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
