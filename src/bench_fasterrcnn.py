#!/usr/bin/env python3
"""
Benchmark torchvision Faster R-CNN on COCO val2017 and evaluate with Ultralytics' COCO evaluator.

Outputs a val_bench.csv-style row per --imgsz containing:
  model, weights, imgsz, speed_preprocess_ms, speed_inference_ms, speed_postprocess_ms, ap50_95, save_dir

Install deps (example):
  pip install ultralytics torchvision pycocotools faster-coco-eval pandas tqdm pillow

COCO folder layout expected:
  COCO_ROOT/
    annotations/instances_val2017.json
    images/val2017/000000000139.jpg
    ...
"""

from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)

# Ultralytics COCO evaluator (reuse their implementation)
try:
    from ultralytics.models.yolo.detect import DetectionValidator
except Exception:  # fallback for older/newer layouts
    from ultralytics.models.yolo.detect.val import DetectionValidator  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--coco-root",
        type=Path,
        required=True,
        help="Path to COCO root (contains val2017/ and annotations/)",
    )
    p.add_argument("--device", type=str, default="cuda:0", help="cuda:0, cpu, etc.")
    p.add_argument(
        "--batch", type=int, default=1, help="Batch size (list-of-images batching)."
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[800],
        help="One or more sizes to benchmark (treated as torchvision min_size / short-side).",
    )
    p.add_argument(
        "--max-size",
        type=int,
        default=0,
        help="Optional explicit torchvision max_size. 0 => imgsz * (1333/800).",
    )
    p.add_argument(
        "--score-thr",
        type=float,
        default=0.001,
        help="Score threshold BEFORE COCO eval. Lower is safer for mAP.",
    )
    p.add_argument(
        "--max-det",
        type=int,
        default=100,
        help="Max detections per image (COCO typically 100).",
    )
    p.add_argument(
        "--warmup", type=int, default=10, help="Warmup iterations for timing."
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only run on first N images (debug).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/val/torchvision_fasterrcnn_resnet50_fpn"),
        help="Output directory root.",
    )
    p.add_argument("--csv-out", type=Path, default=Path("val_bench_torchvision.csv"))
    return p.parse_args()


class CocoVal2017(Dataset):
    """COCO val2017 dataset that also exposes `im_files` for Ultralytics' evaluator."""

    def __init__(self, coco_root: Path, img_transform):
        self.coco_root = coco_root
        self.img_dir = coco_root / "images" / "val2017"
        self.ann_file = coco_root / "annotations" / "instances_val2017.json"
        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"Missing images dir: {self.img_dir}")
        if not self.ann_file.is_file():
            raise FileNotFoundError(f"Missing annotation file: {self.ann_file}")

        self.coco = COCO(str(self.ann_file))
        self.img_ids = sorted(self.coco.getImgIds())
        self.transform = img_transform

        # Ultralytics coco_evaluate() uses stems of these filenames as imgIds.
        # COCO val2017 filenames are like 000000000139.jpg => stem "000000000139" => int 139.
        self.im_files = [
            str(self.img_dir / self.coco.loadImgs(i)[0]["file_name"])
            for i in self.img_ids
        ]

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id = int(self.img_ids[idx])
        info = self.coco.loadImgs(img_id)[0]
        path = self.img_dir / info["file_name"]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)  # float tensor in 0..1
        return img_id, img


def collate_fn(batch):
    img_ids, imgs = zip(*batch)
    return list(img_ids), list(imgs)


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def forward_fasterrcnn_split(
    model,
    images: List[torch.Tensor],
    device: torch.device,
) -> Tuple[List[Dict[str, torch.Tensor]], float, float, float]:
    """
    Run FasterRCNN forward in 3 timed stages:
      preprocess: device transfer + model.transform()  (resize/normalize/pad)
      inference: backbone + rpn + roi_heads (includes NMS)
      postprocess: model.transform.postprocess() (scale boxes back)
    Returns (detections, pre_ms, inf_ms, post_ms) per-batch.
    """
    sync_if_cuda(device)
    t0 = time.perf_counter()

    images = [img.to(device, non_blocking=True) for img in images]
    original_sizes = [tuple(img.shape[-2:]) for img in images]  # [(H,W), ...]

    images_t, _ = model.transform(images, targets=None)
    sync_if_cuda(device)
    t1 = time.perf_counter()

    features = model.backbone(images_t.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    proposals, _ = model.rpn(images_t, features, targets=None)
    detections, _ = model.roi_heads(
        features, proposals, images_t.image_sizes, targets=None
    )
    sync_if_cuda(device)
    t2 = time.perf_counter()

    detections = model.transform.postprocess(
        detections, images_t.image_sizes, original_sizes
    )
    sync_if_cuda(device)
    t3 = time.perf_counter()

    return detections, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3


def build_label_to_coco_catid(weights, coco: COCO) -> Dict[int, int]:
    """
    Map torchvision predicted label indices -> COCO category_id, using category names as bridge.
    This avoids off-by-one / missing-ID issues because COCO category IDs are not contiguous.
    """
    tv_categories = list(
        weights.meta["categories"]
    )  # includes '__background__' and some 'N/A' placeholders
    coco_cats = coco.loadCats(coco.getCatIds())
    name_to_coco_id = {c["name"]: int(c["id"]) for c in coco_cats}

    label_to_catid: Dict[int, int] = {}
    for label_idx, name in enumerate(tv_categories):
        if name == "__background__":
            continue
        coco_id = name_to_coco_id.get(name)
        if coco_id is not None:
            label_to_catid[int(label_idx)] = int(coco_id)
    return label_to_catid


def detections_to_coco_json(
    detections: List[Dict[str, torch.Tensor]],
    img_ids: List[int],
    label_to_catid: Dict[int, int],
    score_thr: float,
    max_det: int,
) -> List[dict]:
    """Convert torchvision detections to COCO results JSON entries."""
    jdict: List[dict] = []
    for det, img_id in zip(detections, img_ids):
        boxes = det["boxes"].detach().cpu()
        scores = det["scores"].detach().cpu()
        labels = det["labels"].detach().cpu()

        if boxes.numel() == 0:
            continue

        order = torch.argsort(scores, descending=True)[:max_det]
        boxes, scores, labels = boxes[order], scores[order], labels[order]

        for box, score, label in zip(boxes, scores, labels):
            s = float(score)
            if s < score_thr:
                break
            cat_id = label_to_catid.get(int(label))
            if cat_id is None:
                continue
            x1, y1, x2, y2 = [float(x) for x in box.tolist()]
            jdict.append(
                {
                    "image_id": int(img_id),
                    "category_id": int(cat_id),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": s,
                }
            )
    return jdict


def pick_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def run_one_imgsz(imgsz: int, args: argparse.Namespace) -> dict:
    device = pick_device(args.device)

    out_dir = args.out_dir / f"imgsz{imgsz}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model + weights
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights).eval().to(device)

    # IMPORTANT: make torchvision internal resize follow `imgsz`
    max_size = (
        int(args.max_size)
        if args.max_size and args.max_size > 0
        else int(round(imgsz * (1333 / 800)))
    )
    model.transform.min_size = (int(imgsz),)  # type: ignore
    model.transform.max_size = int(max_size)  # type: ignore

    # Don’t drop too many predictions before COCO eval
    model.roi_heads.score_thresh = float(args.score_thr)  # type: ignore
    model.roi_heads.detections_per_img = int(args.max_det)  # type: ignore

    dataset = CocoVal2017(args.coco_root, img_transform=weights.transforms())
    if args.limit and args.limit > 0:
        dataset.img_ids = dataset.img_ids[: args.limit]
        dataset.im_files = dataset.im_files[: args.limit]

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    label_to_catid = build_label_to_coco_catid(weights, dataset.coco)

    # Warmup
    if args.warmup and args.warmup > 0:
        dummy = torch.zeros((3, imgsz, imgsz), device=device)
        for _ in range(args.warmup):
            _ = model([dummy])
        sync_if_cuda(device)

    # Inference
    all_preds: List[dict] = []
    pre_ms_sum = inf_ms_sum = post_ms_sum = 0.0
    n_images = 0

    pbar = tqdm(loader, desc=f"imgsz={imgsz}", unit="batch")
    for img_ids, images in pbar:
        dets, pre_ms, inf_ms, post_ms = forward_fasterrcnn_split(model, images, device)

        b = len(images)
        n_images += b
        pre_ms_sum += pre_ms
        inf_ms_sum += inf_ms
        post_ms_sum += post_ms

        # JSON formatting is excluded from the speed buckets (like Ultralytics does)
        all_preds.extend(
            detections_to_coco_json(
                dets,
                img_ids=img_ids,
                label_to_catid=label_to_catid,
                score_thr=args.score_thr,
                max_det=args.max_det,
            )
        )

        pbar.set_postfix(
            pre=f"{pre_ms_sum/n_images:.2f}ms",
            inf=f"{inf_ms_sum/n_images:.2f}ms",
            post=f"{post_ms_sum/n_images:.2f}ms",
        )

    pred_json = out_dir / "predictions.json"
    pred_json.write_text(json.dumps(all_preds))

    # COCO eval via Ultralytics
    # (coco_evaluate only runs if save_json + is_coco + len(jdict) > 0)
    validator = DetectionValidator(
        dataloader=loader, save_dir=out_dir, args={"save_json": True}
    )
    validator.is_coco = True
    validator.is_lvis = False
    validator.jdict = all_preds  # only used for a len() guard
    stats = validator.coco_evaluate({}, pred_json=pred_json, anno_json=dataset.ann_file)  # type: ignore

    ap50_95 = float(stats.get("metrics/mAP50-95(B)", float("nan")))

    return {
        "model": "torchvision_fasterrcnn_resnet50_fpn",
        "weights": str(weights),
        "imgsz": int(imgsz),
        "speed_preprocess_ms": pre_ms_sum / n_images if n_images else float("nan"),
        "speed_inference_ms": inf_ms_sum / n_images if n_images else float("nan"),
        "speed_postprocess_ms": post_ms_sum / n_images if n_images else float("nan"),
        "ap50_95": ap50_95,
        "save_dir": str(out_dir),
    }


def append_row_to_csv(row: dict, csv_path: Path) -> None:
    import pandas as pd

    df_row = pd.DataFrame([row])
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, df_row], ignore_index=True)
    else:
        df = df_row
    df.to_csv(csv_path, index=False)


def main() -> None:
    args = parse_args()
    args.coco_root = args.coco_root.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.csv_out = args.csv_out.expanduser().resolve()

    for imgsz in args.imgsz:
        row = run_one_imgsz(int(imgsz), args)
        append_row_to_csv(row, args.csv_out)
        print(
            f"[OK] imgsz={imgsz} -> ap50_95={row['ap50_95']:.4f} | saved {args.csv_out}"
        )


if __name__ == "__main__":
    main()
