#!/usr/bin/env python3
"""Extract compact visualization-friendly CSVs from Ultralytics benchmark exports."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


REQUIRED_COLUMNS = {
    "model",
    "precision",
    "format_name",
    "imgsz",
    "status",
    "ap50_95",
    "gflops",
    "params",
    "speed_preprocess_ms",
    "speed_inference_ms",
    "speed_postprocess_ms",
}

FORMAT_ALIASES = {
    "pytorch": "PyTorch",
    "pt": "PyTorch",
    "torchscript": "TorchScript",
    "ts": "TorchScript",
    "onnx": "ONNX",
    "tensorrt": "TensorRT",
    "trt": "TensorRT",
}
PRECISION_ALIASES = {
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "fp32": "fp32",
    "float32": "fp32",
    "single": "fp32",
}
LATENCY_ALIASES = {
    "preprocess": "preprocess",
    "pre": "preprocess",
    "speedpreprocessms": "preprocess",
    "inference": "inference",
    "infer": "inference",
    "speedinferencems": "inference",
    "postprocess": "postprocess",
    "post": "postprocess",
    "posprocess": "postprocess",
    "speedpostprocessms": "postprocess",
    "speedposprocessms": "postprocess",
    "endtoend": "end_to_end",
    "e2e": "end_to_end",
    "total": "end_to_end",
    "end2end": "end_to_end",
    "end_to_end": "end_to_end",
    "end-to-end": "end_to_end",
}
LATENCY_SOURCE_COLUMNS = {
    "preprocess": "speed_preprocess_ms",
    "inference": "speed_inference_ms",
    "postprocess": "speed_postprocess_ms",
    "end_to_end": "end_to_end_ms",
}
LATENCY_OUTPUT_COLUMNS = {
    "preprocess": "pre",
    "inference": "inf",
    "postprocess": "post",
    "end_to_end": "e2e",
}


def normalize_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def split_cli_tokens(values: Sequence[str] | None) -> list[str]:
    tokens: list[str] = []
    for value in values or []:
        for token in re.split(r"[,\s]+", value.strip()):
            token = token.strip()
            if token:
                tokens.append(token)
    return tokens


def canonicalize_format(value: str) -> str:
    norm = normalize_key(value)
    if norm in FORMAT_ALIASES:
        return FORMAT_ALIASES[norm]
    return str(value).strip()


def canonicalize_precision(value: str) -> str:
    norm = normalize_key(value)
    if norm in PRECISION_ALIASES:
        return PRECISION_ALIASES[norm]
    return str(value).strip().lower()


def canonicalize_latency_choice(value: str) -> str:
    norm = normalize_key(value)
    if norm in LATENCY_ALIASES:
        return LATENCY_ALIASES[norm]
    choices = ", ".join(sorted(set(LATENCY_ALIASES.values())))
    raise SystemExit(f"Unsupported --latency value: {value!r}. Use one of: {choices}")


def make_setting_slug(precision: str, format_name: str) -> str:
    return f"{precision}-{normalize_key(format_name)}"


def format_scale_label(scale: float) -> str:
    rounded = round(float(scale), 4)
    if math.isclose(rounded, round(rounded), rel_tol=0.0, abs_tol=1e-9):
        return f"{int(round(rounded))}x"
    text = f"{rounded:.4f}".rstrip("0").rstrip(".")
    return f"{text}x"


def parse_models_arg(
    values: Sequence[str] | None, available_models: Sequence[str]
) -> list[str] | None:
    tokens = split_cli_tokens(values)
    if not tokens or any(normalize_key(token) == "all" for token in tokens):
        return None

    available_map = {normalize_key(model): model for model in available_models}
    selected: list[str] = []
    missing: list[str] = []
    for token in tokens:
        key = normalize_key(token)
        if key in available_map:
            selected.append(available_map[key])
        else:
            missing.append(token)
    if missing:
        raise SystemExit(
            "Unknown model(s): "
            + ", ".join(missing)
            + ". Available models: "
            + ", ".join(available_models)
        )
    return list(dict.fromkeys(selected))


def parse_settings_arg(
    values: Sequence[str] | None,
    available_settings: Sequence[tuple[str, str]],
) -> list[tuple[str, str]] | None:
    tokens = split_cli_tokens(values)
    if not tokens or any(normalize_key(token) == "all" for token in tokens):
        return None

    available_set = set(available_settings)
    available_formats = {
        normalize_key(format_name): format_name for _, format_name in available_settings
    }
    available_precisions = {
        normalize_key(precision): precision for precision, _ in available_settings
    }

    selected: list[tuple[str, str]] = []
    errors: list[str] = []

    for token in tokens:
        parts = [
            normalize_key(part) for part in re.split(r"[-/:]", token) if part.strip()
        ]
        if len(parts) < 2:
            errors.append(
                f"{token!r} (expected something like fp32-pytorch or pytorch/fp32)"
            )
            continue

        precision_parts = [
            part
            for part in parts
            if part in PRECISION_ALIASES or part in available_precisions
        ]
        if len(precision_parts) != 1:
            errors.append(
                f"{token!r} (could not determine a single precision; use one of: "
                + ", ".join(sorted(available_precisions.values()))
                + ")"
            )
            continue

        precision_key = precision_parts[0]
        precision = PRECISION_ALIASES.get(
            precision_key, available_precisions[precision_key]
        )

        format_parts = [part for part in parts if part != precision_key]
        format_key = "".join(format_parts)
        format_name = None
        if format_key in available_formats:
            format_name = available_formats[format_key]
        elif format_key in FORMAT_ALIASES:
            candidate = FORMAT_ALIASES[format_key]
            if normalize_key(candidate) in available_formats:
                format_name = available_formats[normalize_key(candidate)]

        if format_name is None:
            errors.append(
                f"{token!r} (unknown format; use one of: "
                + ", ".join(sorted(dict.fromkeys(available_formats.values())))
                + ")"
            )
            continue

        setting = (precision, format_name)
        if setting not in available_set:
            errors.append(f"{token!r} (not available in the input CSV)")
            continue
        selected.append(setting)

    if errors:
        raise SystemExit("Invalid --settings value(s):\n- " + "\n- ".join(errors))
    return list(dict.fromkeys(selected))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract compact, visualization-friendly CSVs from Ultralytics benchmark CSV exports."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python extract_ultralytics_benchmark.py benchmark_ultralytics_jetson.csv jetson_wide.csv\n"
            "  python extract_ultralytics_benchmark.py benchmark_ultralytics_jetson.csv jetson_long.csv --mode long --latency end-to-end --with-std\n"
            "  python extract_ultralytics_benchmark.py benchmark_ultralytics_jetson.csv jetson_selected.csv --settings fp32-pytorch fp16-onnx --models yolo11n yolo11x\n"
            "  python extract_ultralytics_benchmark.py benchmark_ultralytics_jetson.csv outputs/ --split-settings --mode wide\n"
            "  python extract_ultralytics_benchmark.py benchmark_ultralytics_jetson.csv --list-settings\n"
        ),
    )
    parser.add_argument("input_csv", help="Path to the source benchmark CSV.")
    parser.add_argument(
        "output_csv",
        nargs="?",
        help=(
            "Output CSV path. Required unless --list-settings or --list-models is used. "
            "When --split-settings is enabled this can be either a .csv base name or a directory."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("wide", "long"),
        default="wide",
        help="Output shape: 'wide' (deprecated-file style) or tidy 'long'. Default: wide.",
    )
    parser.add_argument(
        "--aggregate",
        choices=("mean", "median"),
        default="mean",
        help="How to collapse repeated runs for the same setting. Default: mean.",
    )
    parser.add_argument(
        "--with-std",
        action="store_true",
        help="Also output standard deviation columns for accuracy and the selected latency metric.",
    )
    parser.add_argument(
        "--include-run-counts",
        action="store_true",
        help="Include total/success/error run counts in the exported CSV.",
    )
    parser.add_argument(
        "--latency",
        default="inference",
        help=(
            "Latency bucket to extract: inference, preprocess, postprocess, or end-to-end. "
            "Aliases like e2e/end2end are also accepted. Default: inference."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional model filter, comma or space separated. Example: --models yolo11n,yolo11x",
    )
    parser.add_argument(
        "--settings",
        nargs="*",
        help=(
            "Optional exact setting filter, comma or space separated. "
            "Examples: --settings fp32-pytorch fp16-onnx OR --settings pytorch/fp32"
        ),
    )
    parser.add_argument(
        "--split-settings",
        action="store_true",
        help=(
            "Write one CSV per (precision, format_name) setting using file names such as "
            "fp32-pytorch and fp16-onnx."
        ),
    )
    parser.add_argument(
        "--native-imgsz",
        type=int,
        default=640,
        help="Image size treated as 1x when computing scale labels. Default: 640.",
    )
    parser.add_argument(
        "--date",
        help="Optional constant date/year value to add as a column to the output CSV(s).",
    )
    parser.add_argument(
        "--list-settings",
        action="store_true",
        help="Print available precision-format settings found in the input and exit.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print available models found in the input and exit.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error summary messages.",
    )
    return parser


def load_and_prepare_dataframe(path: str | Path, native_imgsz: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]

    rename_map = {}
    if "percision" in df.columns and "precision" not in df.columns:
        rename_map["percision"] = "precision"
    if "speed_posprocess_ms" in df.columns and "speed_postprocess_ms" not in df.columns:
        rename_map["speed_posprocess_ms"] = "speed_postprocess_ms"
    if rename_map:
        df = df.rename(columns=rename_map)

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise SystemExit("Input CSV is missing required columns: " + ", ".join(missing))

    df["network"] = df["model"].astype(str).str.strip()
    df["precision"] = df["precision"].astype(str).map(canonicalize_precision)
    df["format_name"] = df["format_name"].astype(str).map(canonicalize_format)
    df["status"] = df["status"].astype(str).str.strip().str.lower()

    df["native_imgsz"] = int(native_imgsz)
    df["native_resolution"] = "1x"
    df["scale"] = df["imgsz"].astype(float) / float(native_imgsz)
    df["scale_label"] = df["scale"].map(format_scale_label)
    df["setting_slug"] = [
        make_setting_slug(precision, format_name)
        for precision, format_name in zip(df["precision"], df["format_name"])
    ]
    df["end_to_end_ms"] = (
        df["speed_preprocess_ms"]
        + df["speed_inference_ms"]
        + df["speed_postprocess_ms"]
    )
    return df


def list_available(df: pd.DataFrame) -> tuple[list[str], list[tuple[str, str]]]:
    models = sorted(df["model"].dropna().astype(str).unique().tolist())
    settings_df = (
        df[["precision", "format_name"]]
        .drop_duplicates()
        .sort_values(["precision", "format_name"])
        .reset_index(drop=True)
    )
    settings = [tuple(row) for row in settings_df.itertuples(index=False, name=None)]
    return models, settings


def filter_dataframe(
    df: pd.DataFrame,
    selected_models: list[str] | None,
    selected_settings: list[tuple[str, str]] | None,
) -> pd.DataFrame:
    result = df
    if selected_models is not None:
        result = result[result["model"].isin(selected_models)]
    if selected_settings is not None:
        mask = pd.Series(False, index=result.index)
        for precision, format_name in selected_settings:
            mask |= (result["precision"] == precision) & (
                result["format_name"] == format_name
            )
        result = result[mask]
    return result.copy()


def reducer_name_to_callable(name: str):
    if name == "mean":
        return "mean"
    if name == "median":
        return "median"
    raise ValueError(f"Unsupported reducer: {name}")


def compute_success_warnings(df: pd.DataFrame) -> list[str]:
    counts = (
        df.groupby(["model", "precision", "format_name", "imgsz"], dropna=False)[
            "status"
        ]
        .agg(
            total_runs="size",
            success_runs=lambda s: int((s == "success").sum()),
            error_runs=lambda s: int((s != "success").sum()),
        )
        .reset_index()
    )
    warnings: list[str] = []
    zero_success = counts[counts["success_runs"] == 0]
    partial_success = counts[
        (counts["success_runs"] > 0) & (counts["success_runs"] < counts["total_runs"])
    ]
    for row in zero_success.itertuples(index=False):
        warnings.append(
            f"No successful runs for model={row.model}, precision={row.precision}, "
            f"format_name={row.format_name}, imgsz={row.imgsz} "
            f"(total={row.total_runs}, errors={row.error_runs})."
        )
    for row in partial_success.itertuples(index=False):
        warnings.append(
            f"Using only {row.success_runs}/{row.total_runs} successful repeats for "
            f"model={row.model}, precision={row.precision}, format_name={row.format_name}, imgsz={row.imgsz}."
        )
    return warnings


def zero_small_values(series: pd.Series, threshold: float = 1e-12) -> pd.Series:
    return series.mask(series.abs() < threshold, 0.0)


def aggregate_to_long(
    df: pd.DataFrame,
    latency_choice: str,
    aggregate: str,
    with_std: bool,
    include_run_counts: bool,
) -> pd.DataFrame:
    latency_source = LATENCY_SOURCE_COLUMNS[latency_choice]
    latency_output = LATENCY_OUTPUT_COLUMNS[latency_choice]
    reducer = reducer_name_to_callable(aggregate)

    success_df = df[df["status"] == "success"].copy()
    if success_df.empty:
        raise SystemExit("No successful rows left after applying filters.")

    group_cols = [
        "model",
        "precision",
        "format_name",
        "native_resolution",
        "native_imgsz",
        "imgsz",
        "scale",
        "scale_label",
    ]

    aggregated = (
        success_df.groupby(group_cols, dropna=False)
        .agg(
            ap50_95=("ap50_95", reducer),
            **{latency_output: (latency_source, reducer)},
            flops=("gflops", reducer),
            params=("params", reducer),
        )
        .reset_index()
    )

    if with_std:
        std_df = (
            success_df.groupby(group_cols, dropna=False)
            .agg(
                ap50_95_std=("ap50_95", lambda s: float(s.std(ddof=0))),
                **{
                    f"{latency_output}_std": (
                        latency_source,
                        lambda s: float(s.std(ddof=0)),
                    )
                },
            )
            .fillna(0.0)
            .reset_index()
        )
        std_df["ap50_95_std"] = zero_small_values(std_df["ap50_95_std"])
        std_df[f"{latency_output}_std"] = zero_small_values(
            std_df[f"{latency_output}_std"]
        )
        aggregated = aggregated.merge(std_df, on=group_cols, how="left")

    if include_run_counts:
        counts_df = (
            df.groupby(group_cols, dropna=False)["status"]
            .agg(
                total_runs="size",
                success_runs=lambda s: int((s == "success").sum()),
                error_runs=lambda s: int((s != "success").sum()),
            )
            .reset_index()
        )
        aggregated = aggregated.merge(counts_df, on=group_cols, how="left")

    aggregated["params"] = aggregated["params"].round().astype(int)
    aggregated = aggregated.sort_values(
        ["model", "precision", "format_name", "imgsz"]
    ).reset_index(drop=True)

    ordered_cols = [
        "model",
        "precision",
        "format_name",
        "native_resolution",
        "native_imgsz",
        "imgsz",
        "scale",
        "scale_label",
        "params",
        "flops",
        "ap50_95",
        latency_output,
    ]
    if with_std:
        ordered_cols.extend(["ap50_95_std", f"{latency_output}_std"])
    if include_run_counts:
        ordered_cols.extend(["total_runs", "success_runs", "error_runs"])
    return aggregated[ordered_cols]


def pivot_metric(
    long_df: pd.DataFrame,
    index_cols: list[str],
    value_col: str,
    prefix: str,
    scale_order: Sequence[str],
) -> pd.DataFrame:
    metric_df = long_df.pivot_table(
        index=index_cols,
        columns="scale_label",
        values=value_col,
        aggfunc="first",
    ).reset_index()
    metric_df.columns.name = None

    existing_scale_cols = [scale for scale in scale_order if scale in metric_df.columns]
    rename_map = {scale: f"{prefix}_{scale}" for scale in existing_scale_cols}
    metric_df = metric_df.rename(columns=rename_map)
    return metric_df


def convert_long_to_wide(
    long_df: pd.DataFrame,
    latency_choice: str,
    with_std: bool,
    include_run_counts: bool,
    date_value: str | None,
) -> pd.DataFrame:
    latency_output = LATENCY_OUTPUT_COLUMNS[latency_choice]
    index_cols = [
        "model",
        "precision",
        "format_name",
        "native_resolution",
        "native_imgsz",
        "params",
    ]
    if date_value is not None:
        long_df = long_df.copy()
        long_df["date"] = date_value
        index_cols = [
            "model",
            "date",
            "precision",
            "format_name",
            "native_resolution",
            "native_imgsz",
            "params",
        ]

    scale_order = (
        long_df[["scale", "scale_label"]]
        .drop_duplicates()
        .sort_values("scale")["scale_label"]
        .tolist()
    )

    wide_df = pivot_metric(long_df, index_cols, "ap50_95", "map", scale_order)
    wide_df = wide_df.merge(
        pivot_metric(long_df, index_cols, latency_output, latency_output, scale_order),
        on=index_cols,
        how="outer",
    )
    wide_df = wide_df.merge(
        pivot_metric(long_df, index_cols, "flops", "flops", scale_order),
        on=index_cols,
        how="outer",
    )

    if with_std:
        wide_df = wide_df.merge(
            pivot_metric(long_df, index_cols, "ap50_95_std", "map_std", scale_order),
            on=index_cols,
            how="outer",
        )
        wide_df = wide_df.merge(
            pivot_metric(
                long_df,
                index_cols,
                f"{latency_output}_std",
                f"{latency_output}_std",
                scale_order,
            ),
            on=index_cols,
            how="outer",
        )

    if include_run_counts:
        for count_col in ["total_runs", "success_runs", "error_runs"]:
            wide_df = wide_df.merge(
                pivot_metric(long_df, index_cols, count_col, count_col, scale_order),
                on=index_cols,
                how="outer",
            )

    id_cols = index_cols.copy()
    metric_cols: list[str] = []
    metric_cols.extend([f"map_{scale}" for scale in scale_order])
    metric_cols.extend([f"{latency_output}_{scale}" for scale in scale_order])
    metric_cols.extend([f"flops_{scale}" for scale in scale_order])

    if with_std:
        metric_cols.extend([f"map_std_{scale}" for scale in scale_order])
        metric_cols.extend([f"{latency_output}_std_{scale}" for scale in scale_order])

    if include_run_counts:
        for count_col in ["total_runs", "success_runs", "error_runs"]:
            metric_cols.extend([f"{count_col}_{scale}" for scale in scale_order])

    existing_cols = [col for col in id_cols + metric_cols if col in wide_df.columns]
    wide_df = (
        wide_df[existing_cols]
        .sort_values([col for col in id_cols if col != "params"] + ["params"])
        .reset_index(drop=True)
    )
    return wide_df


def ensure_output_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_split_output_path(base_output: Path, setting_slug: str) -> Path:
    if base_output.suffix.lower() == ".csv":
        return base_output.with_name(
            f"{base_output.stem}_{setting_slug}{base_output.suffix}"
        )
    return base_output / f"{setting_slug}.csv"


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_output_parent(path)
    df.to_csv(path, index=False)


def print_available_models(models: Sequence[str]) -> None:
    print("Available models:")
    for model in models:
        print(f"- {model}")


def print_available_settings(settings: Sequence[tuple[str, str]]) -> None:
    print("Available settings:")
    for precision, format_name in settings:
        print(
            f"- {make_setting_slug(precision, format_name)}  ({precision}, {format_name})"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    latency_choice = canonicalize_latency_choice(args.latency)
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    df = load_and_prepare_dataframe(input_path, native_imgsz=args.native_imgsz)
    available_models, available_settings = list_available(df)

    if args.list_models:
        print_available_models(available_models)
    if args.list_settings:
        print_available_settings(available_settings)
    if args.list_models or args.list_settings:
        return 0

    if not args.output_csv:
        parser.error(
            "output_csv is required unless --list-settings or --list-models is used."
        )

    selected_models = parse_models_arg(args.models, available_models)
    selected_settings = parse_settings_arg(args.settings, available_settings)

    filtered_df = filter_dataframe(df, selected_models, selected_settings)
    if filtered_df.empty:
        raise SystemExit("No rows left after applying model/setting filters.")

    warnings = compute_success_warnings(filtered_df)
    for message in warnings:
        print(f"WARNING: {message}", file=sys.stderr)

    long_df = aggregate_to_long(
        filtered_df,
        latency_choice=latency_choice,
        aggregate=args.aggregate,
        with_std=args.with_std,
        include_run_counts=args.include_run_counts,
    )
    output_df = (
        long_df
        if args.mode == "long"
        else convert_long_to_wide(
            long_df,
            latency_choice=latency_choice,
            with_std=args.with_std,
            include_run_counts=args.include_run_counts,
            date_value=args.date,
        )
    )

    base_output = Path(args.output_csv)
    if args.split_settings:
        if base_output.suffix.lower() == ".csv":
            base_output.parent.mkdir(parents=True, exist_ok=True)
        else:
            base_output.mkdir(parents=True, exist_ok=True)

        written_paths: list[Path] = []
        for (precision, format_name), subset in output_df.groupby(
            ["precision", "format_name"], dropna=False
        ):
            setting_slug = make_setting_slug(precision, format_name)  # type: ignore
            out_path = resolve_split_output_path(base_output, setting_slug)
            write_csv(subset.reset_index(drop=True), out_path)
            written_paths.append(out_path)
        if not args.quiet:
            print(f"Wrote {len(written_paths)} CSV files:")
            for path in written_paths:
                print(f"- {path}")
    else:
        out_path = base_output
        write_csv(output_df, out_path)
        if not args.quiet:
            print(
                f"Wrote {len(output_df)} rows x {len(output_df.columns)} columns to {out_path}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
