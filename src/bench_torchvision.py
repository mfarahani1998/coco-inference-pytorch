#!/usr/bin/env python3
"""
Benchmark torchvision detection models on COCO val2017 and evaluate with Ultralytics' COCO evaluator.

This mirrors the CSV schema and incremental-writing behavior of bench_ultralytics.py:
  model, weights, imgsz, speed_preprocess_ms, speed_inference_ms, speed_postprocess_ms, ap50_95, save_dir

Resizable torchvision models from the user's original list:
  - fasterrcnn_resnet50_fpn
  - retinanet_resnet50_fpn
  - maskrcnn_resnet50_fpn

NOT resizable (fixed 320x320 input by design):
  - ssdlite320_mobilenet_v3_large

Notes on "imgsz":
  - For Faster/Mask R-CNN and RetinaNet, torchvision internally resizes inputs using GeneralizedRCNNTransform
    (min_size/max_size). This script sets model.transform.min_size=(imgsz,) and model.transform.max_size.
    That means imgsz is the *short-side* target, not necessarily a square resize.
  - If you want a forced square resize, use --resize-mode fixed (warps to imgsz x imgsz).

Requirements:
  pip install torchvision torch ultralytics pycocotools faster-coco-eval pyyaml tqdm pillow
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import yaml
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    retinanet_resnet50_fpn,
    RetinaNet_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)

# Ultralytics COCO evaluator
try:
    from ultralytics.models.yolo.detect.val import DetectionValidator
except Exception:  # pragma: no cover
    # Fallback for some Ultralytics layouts
    from ultralytics.models.yolo.detect import DetectionValidator  # type: ignore


# -----------------------------
# Model registry
# -----------------------------

RESIZABLE_TORCHVISION_MODELS = [
    "fasterrcnn_resnet50_fpn",
    "retinanet_resnet50_fpn",
    "maskrcnn_resnet50_fpn",
]

FIXED_SIZE_TORCHVISION_MODELS = {
    "ssdlite320_mobilenet_v3_large": 320,
}

MODEL_ZOO = {
    "fasterrcnn_resnet50_fpn": (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
    ),
    "retinanet_resnet50_fpn": (retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights),
    "maskrcnn_resnet50_fpn": (maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights),
    "ssdlite320_mobilenet_v3_large": (
        ssdlite320_mobilenet_v3_large,
        SSDLite320_MobileNet_V3_Large_Weights,
    ),
}


@dataclass(frozen=True)
class RunSpec:
    model: str
    imgsz: int


# -----------------------------
# COCO dataset helper
# -----------------------------


def resolve_coco_from_ultralytics_yaml(data_yaml: Path) -> Tuple[Path, Path]:
    """
    Minimal parser for an Ultralytics-style coco.yaml.

    Returns:
      img_dir: directory that contains val images (e.g., .../images/val2017)
      anno_json: path to instances_val2017.json
    """
    data_yaml = data_yaml.expanduser().resolve()
    d = yaml.safe_load(data_yaml.read_text())

    # Ultralytics uses 'path' as dataset root
    root = Path(d.get("path"))
    if not root.is_absolute():
        root = (data_yaml.parent.parent / root).resolve()

    val = d.get("val")
    if val is None:
        raise KeyError(f"{data_yaml} missing required key: val")

    val_path = Path(val)
    if not val_path.is_absolute():
        val_path = (root / val_path).resolve()

    # COCO annotations are typically here in Ultralytics datasets
    anno = root / "annotations" / "instances_val2017.json"
    if not anno.is_file():
        # fallback: try to find something sensible
        ann_dir = root / "annotations"
        if ann_dir.is_dir():
            cand = sorted(ann_dir.glob("instances_val*.json"))
            if cand:
                anno = cand[0]
        if not anno.is_file():
            raise FileNotFoundError(
                f"Could not find instances_val*.json under {root}/annotations"
            )

    if val_path.is_file() and val_path.suffix == ".txt":
        # Ultralytics sometimes uses val2017.txt. We need the image dir for COCO API,
        # so infer it from the first line.
        lines = [x.strip() for x in val_path.read_text().splitlines() if x.strip()]
        if not lines:
            raise ValueError(f"Empty val txt file: {val_path}")
        first = Path(lines[0])
        img_dir = first.parent
    else:
        img_dir = val_path

    if not img_dir.is_absolute():
        img_dir = (root / img_dir).resolve()

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Validation images dir not found: {img_dir}")

    return img_dir, anno


class CocoValDataset(Dataset):
    """
    COCO val dataset that returns (img_id, img_tensor) and exposes `.im_files` for Ultralytics coco_evaluate().
    """

    def __init__(self, img_dir: Path, anno_json: Path, img_transform):
        self.img_dir = img_dir
        self.anno_json = anno_json
        self.coco = COCO(str(anno_json))
        self.img_ids = sorted(self.coco.getImgIds())
        self.transform = img_transform

        # Ultralytics coco_evaluate reads stems of these paths to build imgIds list.
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
        img = self.transform(img)
        return img_id, img


def collate_fn(batch):
    img_ids, imgs = zip(*batch)
    return list(img_ids), list(imgs)


# -----------------------------
# Timing helpers
# -----------------------------


def pick_device(device: str | int) -> torch.device:
    # Accept Ultralytics-style device values (e.g. 0, "0", "cuda:0", "cpu")
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    if isinstance(device, int):
        device = f"cuda:{device}"
    device = str(device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def forward_generalized_rcnn_split(
    model,
    images: List[torch.Tensor],
    device: torch.device,
) -> Tuple[List[Dict[str, torch.Tensor]], float, float, float]:
    """
    For FasterRCNN / MaskRCNN (GeneralizedRCNN subclasses):
      preprocess: device transfer + model.transform()
      inference: backbone + rpn + roi_heads (includes NMS)
      postprocess: model.transform.postprocess() (scale boxes back)
    """
    sync_if_cuda(device)
    t0 = time.perf_counter()

    images = [img.to(device, non_blocking=True) for img in images]
    original_sizes = [tuple(img.shape[-2:]) for img in images]
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


@torch.inference_mode()
def forward_retinanet_split(
    model,
    images: List[torch.Tensor],
    device: torch.device,
) -> Tuple[List[Dict[str, torch.Tensor]], float, float, float]:
    """
    For torchvision RetinaNet:
      preprocess: device transfer + model.transform()
      inference: backbone + head + anchor gen + postprocess_detections (includes NMS)
      postprocess: model.transform.postprocess() (scale boxes back)
    """
    sync_if_cuda(device)
    t0 = time.perf_counter()

    images = [img.to(device, non_blocking=True) for img in images]
    original_sizes: List[Tuple[int, int]] = [tuple(img.shape[-2:]) for img in images]  # type: ignore
    images_t, _ = model.transform(images, targets=None)

    sync_if_cuda(device)
    t1 = time.perf_counter()

    features = model.backbone(images_t.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    features_list = list(features.values())

    head_outputs = model.head(features_list)
    anchors = model.anchor_generator(images_t, features_list)

    # Recover level sizes and split outputs per level (from torchvision RetinaNet.forward)
    num_anchors_per_level = [x.size(2) * x.size(3) for x in features_list]
    HW = 0
    for v in num_anchors_per_level:
        HW += v
    HWA = head_outputs["cls_logits"].size(1)
    A = HWA // HW
    num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

    split_head_outputs: Dict[str, List[torch.Tensor]] = {}
    for k in head_outputs:
        split_head_outputs[k] = list(
            head_outputs[k].split(num_anchors_per_level, dim=1)
        )
    split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

    detections = model.postprocess_detections(
        split_head_outputs, split_anchors, images_t.image_sizes
    )

    sync_if_cuda(device)
    t2 = time.perf_counter()

    detections = model.transform.postprocess(
        detections, images_t.image_sizes, original_sizes
    )

    sync_if_cuda(device)
    t3 = time.perf_counter()

    return detections, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3


# -----------------------------
# COCO JSON conversion + eval
# -----------------------------


def build_label_to_coco_catid(weights, coco: COCO) -> Dict[int, int]:
    """
    Map torchvision predicted label indices -> COCO category_id by matching category names.
    This avoids mismatches because COCO category IDs are not contiguous.
    """
    tv_categories = list(weights.meta["categories"])
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


def coco_map_via_ultralytics(
    dataloader: DataLoader,
    save_dir: Path,
    pred_json: Path,
    anno_json: Path,
    jdict: List[dict],
) -> Dict[str, Any]:
    """
    Run Ultralytics faster-coco-eval through DetectionValidator.coco_evaluate().

    Comparability/robustness improvements:
      - Enforces Path inputs (Ultralytics calls .is_file()).
      - Enforces dataset.im_files presence (Ultralytics builds imgIds from it).
      - Filters jdict to exactly the evaluated imgIds (important for --limit/subsets).
      - Forces is_coco/save_json guard flags explicitly.
    """
    save_dir = Path(save_dir)
    pred_json = Path(pred_json)
    anno_json = Path(anno_json)

    if not pred_json.is_file():
        raise FileNotFoundError(f"pred_json not found: {pred_json}")
    if not anno_json.is_file():
        raise FileNotFoundError(f"anno_json not found: {anno_json}")
    if not jdict:
        raise ValueError("Empty jdict: no predictions to evaluate.")

    # Ultralytics coco_evaluate uses stems of these paths to build imgIds:
    # val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]
    im_files = getattr(dataloader.dataset, "im_files", None)
    if not im_files:
        raise AttributeError(
            "dataloader.dataset must expose `im_files` (list of image paths). "
            "Ultralytics coco_evaluate derives imgIds from these stems."
        )

    try:
        eval_img_ids = [int(Path(p).stem) for p in im_files]
    except Exception as e:
        raise ValueError(
            "Failed to parse int image_ids from dataset.im_files stems. "
            "For COCO val2017, filenames should be like 000000000139.jpg."
        ) from e

    eval_id_set = set(eval_img_ids)

    # Filter predictions to exactly the images that will be evaluated.
    # (Extra preds are typically ignored by COCOeval when imgIds is set,
    #  but filtering avoids confusing edge cases and makes subset runs cleaner.)
    filtered = []
    for p in jdict:
        try:
            iid = int(p.get("image_id", -1))
        except Exception:
            continue
        if iid in eval_id_set:
            filtered.append(p)

    if not filtered:
        raise ValueError(
            "After filtering predictions to evaluated imgIds, jdict became empty. "
            "This usually means your prediction image_id mapping does not match "
            "dataset.im_files stems."
        )

    v = DetectionValidator(
        dataloader=dataloader, save_dir=save_dir, args={"save_json": True}
    )

    # Satisfy coco_evaluate guard:
    v.is_coco = True
    v.is_lvis = False
    v.jdict = filtered

    # Some Ultralytics versions store args differently; be defensive
    try:
        v.args.save_json = True
    except Exception:
        pass

    stats: Dict[str, Any] = {}
    # Use Ultralytics defaults for full comparability (iou_types="bbox", suffix="Box" -> "(B)" keys)
    stats = v.coco_evaluate(stats, pred_json=pred_json, anno_json=anno_json)  # type: ignore
    return stats


# -----------------------------
# Benchmark runner
# -----------------------------


def build_model(model_name: str, device: torch.device, score_thr: float, max_det: int):
    if model_name not in MODEL_ZOO:
        raise KeyError(f"Unknown model: {model_name}. Known: {list(MODEL_ZOO)}")

    ctor, WeightsEnum = MODEL_ZOO[model_name]
    weights = WeightsEnum.DEFAULT
    model = ctor(weights=weights).eval().to(device)

    # Set detection thresholds to keep enough predictions for COCO eval
    if model_name in ("fasterrcnn_resnet50_fpn", "maskrcnn_resnet50_fpn"):
        model.roi_heads.score_thresh = float(score_thr)  # type: ignore[attr-defined]
        model.roi_heads.detections_per_img = int(max_det)  # type: ignore[attr-defined]
    elif model_name == "retinanet_resnet50_fpn":
        model.score_thresh = float(score_thr)  # type: ignore[attr-defined]
        model.detections_per_img = int(max_det)  # type: ignore[attr-defined]
    elif model_name == "ssdlite320_mobilenet_v3_large":
        model.score_thresh = float(score_thr)  # type: ignore[attr-defined]
        model.detections_per_img = int(max_det)  # type: ignore[attr-defined]

    return model, weights


def set_torchvision_resize(
    model, model_name: str, imgsz: int, max_size: int, resize_mode: str
) -> None:
    """
    Goal: maximize comparability with Ultralytics `imgsz` for val/predict.

    Ultralytics (rect=True, batch=1) effectively:
      - scales so max(H,W) ~= imgsz (preserve aspect)
      - applies minimal padding to stride (not full square padding)

    Torchvision equivalent (without editing model code):
      - set GeneralizedRCNNTransform so that long-side is capped at imgsz:
            min_size = imgsz
            max_size = imgsz

    resize_mode:
      - "short": (REVISED FOR COMPARABILITY) cap long-side at imgsz (aspect-preserving)
      - "fixed": force warp to (imgsz,imgsz) via fixed_size (NOT letterbox; generally not comparable)
      - "none": keep torchvision defaults (useful to reproduce official torchvision AP numbers)
    """
    if model_name in FIXED_SIZE_TORCHVISION_MODELS:
        return  # truly fixed; nothing to set here

    if resize_mode == "none":
        return

    t = getattr(model, "transform", None)
    if t is None:
        raise AttributeError(
            f"{model_name} has no .transform; cannot set resize behavior."
        )

    # Always clear fixed_size unless explicitly requested
    if hasattr(t, "fixed_size") and t.fixed_size is not None and resize_mode != "fixed":
        t.fixed_size = None

    # Match Ultralytics imgsz semantics: fit within imgsz with max side capped at imgsz.
    if resize_mode == "short":
        # IMPORTANT: ignore max_size_eff ratios (e.g. 1333/800). Those give torchvision *more pixels*
        # than Ultralytics at the same imgsz and hurt comparability.
        t.min_size = (int(imgsz),)
        t.max_size = int(imgsz)

        # Ensure stride-style padding behavior matches common YOLO stride (32).
        # (Torchvision defaults are usually already 32, but set explicitly for stability.)
        if hasattr(t, "size_divisible"):
            t.size_divisible = 32
        return

    if resize_mode == "fixed":
        # WARNING: This warps aspect ratio. It is NOT the same as YOLO letterbox-to-square.
        # Use only if you intentionally want "warp-to-square" experiments.
        t.fixed_size = (int(imgsz), int(imgsz))
        t.min_size = (int(imgsz),)
        t.max_size = int(imgsz)
        if hasattr(t, "size_divisible"):
            t.size_divisible = 32
        return

    raise ValueError(f"Unknown resize_mode: {resize_mode}")


def run_one(
    spec: RunSpec,
    img_dir: Path,
    anno_json: Path,
    batch: int,
    device: str | int,
    project: str,
    exist_ok: bool,
    workers: int,
    warmup: int,
    score_thr: float,
    max_det: int,
    max_size: int,
    resize_mode: str,
    limit: int = 0,
) -> Dict[str, Any]:
    device_t = pick_device(device)

    model_name = spec.model
    imgsz = spec.imgsz

    # Enforce fixed-size models (optional)
    if model_name in FIXED_SIZE_TORCHVISION_MODELS:
        fixed = FIXED_SIZE_TORCHVISION_MODELS[model_name]
        if imgsz != fixed:
            raise ValueError(
                f"{model_name} has fixed input size {fixed}x{fixed}; got imgsz={imgsz}"
            )

    # Use deterministic run names so your folders don't become val11, val12, ...
    run_name = f"torchvision_{model_name}_img{imgsz}"
    save_dir = Path(project).expanduser().resolve() / run_name
    if save_dir.exists() and not exist_ok:
        raise FileExistsError(f"{save_dir} exists and --exist-ok is false")
    save_dir.mkdir(parents=True, exist_ok=True)

    model, weights = build_model(
        model_name, device_t, score_thr=score_thr, max_det=max_det
    )

    # Set torchvision internal resize behavior
    if max_size <= 0:
        # Default ratio used by torchvision presets is 1333/800 for many detection models.
        # Keep that ratio if the user doesn't specify max_size.
        max_size_eff = int(round(imgsz * (1333 / 800)))
    else:
        max_size_eff = int(max_size)
    set_torchvision_resize(
        model, model_name, imgsz=imgsz, max_size=max_size_eff, resize_mode=resize_mode
    )

    # Dataset + loader (use model weights' recommended transforms: typically just ToTensor/scale to 0..1)
    dataset = CocoValDataset(
        img_dir=img_dir, anno_json=anno_json, img_transform=weights.transforms()
    )
    if limit and limit > 0:
        dataset.img_ids = dataset.img_ids[:limit]
        dataset.im_files = dataset.im_files[:limit]

    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device_t.type == "cuda"),
        collate_fn=collate_fn,
    )

    label_to_catid = build_label_to_coco_catid(weights, dataset.coco)

    # Warmup (measurements below exclude warmup)
    if warmup and warmup > 0:
        dummy = torch.zeros((3, imgsz, imgsz), device=device_t)
        for _ in range(warmup):
            _ = model([dummy])
        sync_if_cuda(device_t)

    # Inference loop
    all_preds: List[dict] = []
    pre_ms_sum = inf_ms_sum = post_ms_sum = 0.0
    n_images = 0

    pbar = tqdm(loader, desc=run_name, unit="batch")
    for img_ids, images in pbar:
        b = len(images)
        if model_name in ("fasterrcnn_resnet50_fpn", "maskrcnn_resnet50_fpn"):
            dets, pre_ms, inf_ms, post_ms = forward_generalized_rcnn_split(
                model, images, device_t
            )
        elif model_name == "retinanet_resnet50_fpn":
            dets, pre_ms, inf_ms, post_ms = forward_retinanet_split(
                model, images, device_t
            )
        elif model_name == "ssdlite320_mobilenet_v3_large":
            # SSD uses a different internal implementation; for simplicity treat whole forward as "inference"
            # and count device transfer as preprocess.
            sync_if_cuda(device_t)
            t0 = time.perf_counter()
            images_dev = [im.to(device_t, non_blocking=True) for im in images]
            sync_if_cuda(device_t)
            t1 = time.perf_counter()
            dets = model(images_dev)  # type: ignore[call-arg]
            sync_if_cuda(device_t)
            t2 = time.perf_counter()
            pre_ms, inf_ms, post_ms = (t1 - t0) * 1e3, (t2 - t1) * 1e3, 0.0
        else:
            raise KeyError(model_name)

        n_images += b
        pre_ms_sum += pre_ms
        inf_ms_sum += inf_ms
        post_ms_sum += post_ms

        all_preds.extend(
            detections_to_coco_json(
                dets,
                img_ids=img_ids,
                label_to_catid=label_to_catid,
                score_thr=score_thr,
                max_det=max_det,
            )
        )

        pbar.set_postfix(
            pre=f"{pre_ms_sum/n_images:.2f}ms",
            inf=f"{inf_ms_sum/n_images:.2f}ms",
            post=f"{post_ms_sum/n_images:.2f}ms",
        )

    pred_json = save_dir / "predictions.json"
    pred_json.write_text(json.dumps(all_preds))

    stats = coco_map_via_ultralytics(
        dataloader=loader,
        save_dir=save_dir,
        pred_json=pred_json,
        anno_json=anno_json,
        jdict=all_preds,
    )
    ap50_95 = float(stats.get("metrics/mAP50-95(B)", float("nan")))

    return dict(
        model=model_name,
        weights=f"torchvision::{weights.__class__.__name__}.{weights.name}",
        imgsz=imgsz,
        speed_preprocess_ms=pre_ms_sum / n_images if n_images else float("nan"),
        speed_inference_ms=inf_ms_sum / n_images if n_images else float("nan"),
        speed_postprocess_ms=post_ms_sum / n_images if n_images else float("nan"),
        ap50_95=ap50_95,
        save_dir=str(save_dir),
    )


FIELDNAMES = [
    "model",
    "weights",
    "imgsz",
    "speed_preprocess_ms",
    "speed_inference_ms",
    "speed_postprocess_ms",
    "ap50_95",
    "save_dir",
]


def write_csv(rows: Iterable[Dict[str, Any]], out_csv: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    out_csv = out_csv.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow(r)
        f.flush()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        type=Path,
        default=Path("configs/coco.yaml"),
        help="Ultralytics-style dataset YAML (default: configs/coco.yaml)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=RESIZABLE_TORCHVISION_MODELS,
        help=f"Models to run. Default: {RESIZABLE_TORCHVISION_MODELS}",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[384, 512, 640, 768],
        help="One or more imgsz values (short-side unless --resize-mode fixed).",
    )
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--device", default="0", help='0, "cuda:0", or "cpu"')
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--project",
        type=str,
        default="runs/val_bench",
        help="Output root folder (like Ultralytics project).",
    )
    p.add_argument("--out-csv", type=Path, default=Path("val_bench.csv"))
    p.add_argument(
        "--exist-ok",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing run folders (default True).",
    )
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--score-thr", type=float, default=0.001)
    p.add_argument("--max-det", type=int, default=100)
    p.add_argument(
        "--max-size",
        type=int,
        default=0,
        help="Torchvision max_size. 0 => imgsz * (1333/800).",
    )
    p.add_argument(
        "--resize-mode",
        choices=["short", "fixed", "none"],
        default="fixed",
        help="How to interpret imgsz for torchvision models.",
    )
    p.add_argument(
        "--limit", type=int, default=0, help="Debug: limit number of images evaluated."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    img_dir, anno_json = resolve_coco_from_ultralytics_yaml(args.data)

    rows = []
    for m in args.models:
        if m not in MODEL_ZOO:
            print(f"[WARN] Unknown model '{m}'. Known: {list(MODEL_ZOO)}")
            continue

        # Optionally skip fixed-size models when user wants resizable list only
        if m in FIXED_SIZE_TORCHVISION_MODELS and args.resize_mode != "none":
            print(
                f"[INFO] {m} is fixed-size ({FIXED_SIZE_TORCHVISION_MODELS[m]}x{FIXED_SIZE_TORCHVISION_MODELS[m]})."
            )

        for s in args.imgsz:
            spec = RunSpec(model=m, imgsz=int(s))
            row = run_one(
                spec,
                img_dir=img_dir,
                anno_json=anno_json,
                batch=int(args.batch),
                device=args.device,
                project=args.project,
                exist_ok=bool(args.exist_ok),
                workers=int(args.workers),
                warmup=int(args.warmup),
                score_thr=float(args.score_thr),
                max_det=int(args.max_det),
                max_size=int(args.max_size),
                resize_mode=str(args.resize_mode),
                limit=int(args.limit),
            )

            print(
                f"{row['model']:>28} imgsz={row['imgsz']:>4} | "
                f"pre={row['speed_preprocess_ms']:.4f}ms "
                f"inf={row['speed_inference_ms']:.4f}ms "
                f"post={row['speed_postprocess_ms']:.4f}ms | "
                f"AP50-95={row['ap50_95']:.6f}"
            )

            rows.append(row)
            write_csv([row], args.out_csv)

    print(f"\nSaved summary to: {Path(args.out_csv).expanduser().resolve()}")


if __name__ == "__main__":
    main()
