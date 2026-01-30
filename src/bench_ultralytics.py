#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterable

# Validators (these return the stats dict we want)
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.models.rtdetr.val import RTDETRValidator


@dataclass(frozen=True)
class RunSpec:
    weights: Path
    imgsz: int


def pick_validator(weights_path: Path):
    stem = weights_path.stem.lower()
    if "rtdetr" in stem or "rt-detr" in stem:
        return RTDETRValidator
    return DetectionValidator


def run_one(
    spec: RunSpec,
    data_yaml: str,
    batch: int,
    device: str | int,
    project: str,
    exist_ok: bool = True,
) -> Dict[str, Any]:
    """
    Runs Ultralytics validation and returns:
      - speed preprocess/inference/postprocess (ms per image, float)
      - AP50-95 from faster-coco-eval (stats['metrics/mAP50-95(B)'])
    """
    Validator = pick_validator(spec.weights)

    # Use deterministic run names so your folders don't become val11, val12, ...
    run_name = f"{spec.weights.stem}_img{spec.imgsz}"

    args = dict(
        mode="val",
        model=str(spec.weights),
        data=data_yaml,
        imgsz=spec.imgsz,
        batch=batch,
        device=device,
        rect=True,  # matches Model.val() default behavior
        plots=False,
        verbose=False,  # avoids per-class spam
        project=project,
        name=run_name,
        exist_ok=exist_ok,
        # save_json is auto-enabled on COCO val2017 in Ultralytics detect validator,
        # but we set it explicitly to be safe:
        save_json=True,
    )

    v = Validator(args=args)

    # IMPORTANT: this returns the stats dict (and gets updated by eval_json -> faster-coco-eval)
    stats: Dict[str, Any] = v()

    # v.speed is full precision floats; stdout prints only 1 decimal. :contentReference[oaicite:5]{index=5}
    sp = v.speed
    pre_ms = float(sp.get("preprocess", float("nan")))
    inf_ms = float(sp.get("inference", float("nan")))
    post_ms = float(sp.get("postprocess", float("nan")))

    # faster-coco-eval "first row" is stored here. :contentReference[oaicite:6]{index=6}
    ap5095 = float(stats.get("metrics/mAP50-95(B)", float("nan")))

    return dict(
        model=spec.weights.stem,
        weights=str(spec.weights),
        imgsz=spec.imgsz,
        speed_preprocess_ms=pre_ms,
        speed_inference_ms=inf_ms,
        speed_postprocess_ms=post_ms,
        ap50_95=ap5095,
        save_dir=str(getattr(v, "save_dir", "")),
    )


def write_csv(rows: Iterable[Dict[str, Any]], out_csv: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    new_file = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow(r)
        f.flush()


def main() -> None:
    models = [
        "yolo26n",
        "yolo26m",
        "yolo26x",
        "rtdetr-l",
        "rtdetr-x",
        "yolov5nu",
        "yolov5mu",
        "yolov5xu",
    ]
    sizes = [384, 512, 640, 768]

    weights_dir = Path("weights")
    data_yaml = "configs/coco.yaml"
    batch = 1
    device = 0  # or "cuda:0"
    project = "runs/val_bench"
    out_csv = Path("val_bench.csv")

    rows = []
    for m in models:
        for s in sizes:
            spec = RunSpec(weights=weights_dir / f"{m}.pt", imgsz=s)
            if not spec.weights.exists():
                print(f"[WARN] Missing weights: {spec.weights}")
                continue
            row = run_one(
                spec,
                data_yaml=data_yaml,
                batch=batch,
                device=device,
                project=project,
                exist_ok=True,
            )

            # Print with higher precision than Ultralytics stdout
            print(
                f"{row['model']:>8} imgsz={row['imgsz']:>4} | "
                f"pre={row['speed_preprocess_ms']:.4f}ms "
                f"inf={row['speed_inference_ms']:.4f}ms "
                f"post={row['speed_postprocess_ms']:.4f}ms | "
                f"AP50-95={row['ap50_95']:.6f}"
            )
            rows.append(row)

            # Write incrementally so you don't lose results if something crashes mid-run
            write_csv([row], out_csv)

    print(f"\nSaved summary to: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
