#!/usr/bin/env python3
"""
Benchmark torchvision detection models on COCO val2017 with a benchmark layout that is
as close as practical to Ultralytics benchmark mode:
  - fixed square imgsz per run
  - low confidence threshold (0.001)
  - max_det=100
  - inference-only timing
  - COCO mAP50-95 computed via Ultralytics/faster-coco-eval outside the timed region
  - multiple independent repeats without aggregation

Backends implemented:
  - PyTorch eager
  - TorchScript
  - ONNX (ONNX Runtime)
  - TensorRT engine (ONNX -> trtexec -> TensorRT runtime)

Important geometry note:
Torchvision detection models do not have a built-in benchmark helper equivalent to
Ultralytics' model.benchmark(). To keep exported formats comparable, this script uses a
single external aspect-preserving letterbox-to-square preprocessing step for all
Torchvision backends, then runs model inference on that fixed square tensor. Prediction
conversion back to original image coordinates happens after the timed region.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools.coco import COCO
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
)

try:
    from ultralytics.models.yolo.detect.val import DetectionValidator
except Exception:  # pragma: no cover
    from ultralytics.models.yolo.detect import DetectionValidator  # type: ignore


DEFAULT_MODELS = [
    "fasterrcnn_resnet50_fpn",
    "retinanet_resnet50_fpn",
    "maskrcnn_resnet50_fpn",
]
DEFAULT_SIZES = [384, 512, 640, 768]
DEFAULT_FORMATS = [
    "pytorch",
    # "torchscript",
    # "onnx",
    # "trt",
]
FIELDNAMES = [
    "framework",
    "model",
    "weights",
    "imgsz",
    "batch",
    "device",
    "repeat",
    "half",
    "precision",
    "format_name",
    "format_arg",
    "backend",
    "runtime_provider",
    "benchmark_impl",
    "input_geometry",
    "resize_mode",
    "conf",
    "max_det",
    "speed_preprocess_ms",
    "speed_inference_ms",
    "speed_postprocess_ms",
    "ap50_95",
    "fps",
    "status",
    "error",
    "artifact_path",
    "artifact_size_mb",
    "save_dir",
    "run_name",
    "metric_key",
    "benchmark_wall_time_s",
]
MODEL_ZOO = {
    "fasterrcnn_resnet50_fpn": (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
    ),
    "retinanet_resnet50_fpn": (
        retinanet_resnet50_fpn,
        RetinaNet_ResNet50_FPN_Weights,
    ),
    "maskrcnn_resnet50_fpn": (
        maskrcnn_resnet50_fpn,
        MaskRCNN_ResNet50_FPN_Weights,
    ),
}
FORMAT_TO_NAME = {
    "pytorch": "PyTorch",
    "torchscript": "TorchScript",
    "onnx": "ONNX",
    "trt": "TensorRT",
}
FORMAT_TO_ARG = {
    "pytorch": "-",
    "torchscript": "torchscript",
    "onnx": "onnx",
    "trt": "engine",
}
METRIC_KEY = "metrics/mAP50-95(B)"
try:
    PIL_BILINEAR = Image.Resampling.BILINEAR  # Pillow >= 9.1
except AttributeError:  # pragma: no cover
    PIL_BILINEAR = Image.BILINEAR  # type: ignore


@dataclass(frozen=True)
class RunSpec:
    model: str
    imgsz: int
    format_key: str
    half: bool
    repeat: int


class CocoLetterboxDataset(Dataset):
    """COCO val dataset with external fixed-square letterbox preprocessing."""

    def __init__(self, img_dir: Path, anno_json: Path, imgsz: int) -> None:
        self.img_dir = img_dir
        self.anno_json = anno_json
        self.imgsz = int(imgsz)
        self.coco = COCO(str(anno_json))
        self.img_ids = sorted(self.coco.getImgIds())
        self.im_files = [
            str(self.img_dir / self.coco.loadImgs(image_id)[0]["file_name"])
            for image_id in self.img_ids
        ]

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor, Dict[str, Any]]:
        image_id = int(self.img_ids[index])
        info = self.coco.loadImgs(image_id)[0]
        path = self.img_dir / info["file_name"]
        image = Image.open(path).convert("RGB")
        tensor, meta = letterbox_pil_to_tensor(image, self.imgsz)
        return image_id, tensor, meta


def collate_fn(
    batch: Sequence[Tuple[int, torch.Tensor, Dict[str, Any]]],
) -> Tuple[List[int], torch.Tensor, List[Dict[str, Any]]]:
    image_ids = [item[0] for item in batch]
    images = torch.stack([item[1] for item in batch], dim=0)
    metas = [item[2] for item in batch]
    return image_ids, images, metas


class TVDetExportWrapper(nn.Module):
    """Wrap torchvision detectors into a fixed-output tensor interface for export/runtime parity."""

    def __init__(self, model: nn.Module, max_det: int) -> None:
        super().__init__()
        self.model = model
        self.max_det = int(max_det)

    def forward(
        self, batch_images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch_images.dim() == 3:
            batch_images = batch_images.unsqueeze(0)

        images = []  # type: List[torch.Tensor]
        for i in range(batch_images.shape[0]):
            images.append(batch_images[i])

        outputs = self.model(images)
        batch_size = batch_images.shape[0]
        boxes = batch_images.new_zeros((batch_size, self.max_det, 4))
        scores = batch_images.new_zeros((batch_size, self.max_det))
        labels = torch.zeros(
            (batch_size, self.max_det), dtype=torch.int32, device=batch_images.device
        )
        num_det = torch.zeros(
            (batch_size,), dtype=torch.int32, device=batch_images.device
        )

        for batch_index, det in enumerate(outputs):
            det_boxes = det["boxes"]
            det_scores = det["scores"]
            det_labels = det["labels"]
            n = min(int(det_boxes.shape[0]), self.max_det)
            if n <= 0:
                continue
            boxes[batch_index, :n] = det_boxes[:n]
            scores[batch_index, :n] = det_scores[:n].to(dtype=scores.dtype)
            labels[batch_index, :n] = det_labels[:n].to(dtype=torch.int32)
            num_det[batch_index] = int(n)

        return boxes, scores, labels, num_det


class OnnxRuntimeRunner(object):
    def __init__(
        self,
        onnx_path: Path,
        device: torch.device,
        imgsz: int,
        max_det: int,
        half: bool,
    ) -> None:
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise RuntimeError("Failed to import onnxruntime: %s" % exc)

        self.ort = ort
        self.onnx_path = onnx_path
        self.device = device
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)
        self.half = bool(half)
        self.dtype_np = np.float16 if self.half else np.float32
        providers = ["CPUExecutionProvider"]
        if device.type == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.providers = list(self.session.get_providers())
        self.provider = self.providers[0] if self.providers else ""
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.io_binding = None
        self.input_ortvalue = None
        self.output_ortvalues = {}  # type: Dict[str, Any]
        self.output_shapes = {
            "boxes": [1, self.max_det, 4],
            "scores": [1, self.max_det],
            "labels": [1, self.max_det],
            "num_det": [1],
        }

        if self.device.type == "cuda" and self.provider == "CUDAExecutionProvider":
            device_id = cuda_device_index(device)
            self.io_binding = self.session.io_binding()
            self.input_ortvalue = ort.OrtValue.ortvalue_from_shape_and_type(
                [1, 3, self.imgsz, self.imgsz], self.dtype_np, "cuda", device_id
            )
            self.io_binding.bind_input(
                name=self.input_name,
                device_type="cuda",
                device_id=device_id,
                element_type=self.dtype_np,
                shape=self.input_ortvalue.shape(),
                buffer_ptr=self.input_ortvalue.data_ptr(),
            )
            for output_name in self.output_names:
                output_shape = self.output_shapes.get(output_name)
                if output_shape is None:
                    raise RuntimeError("Unexpected ONNX output name: %s" % output_name)
                output_dtype = (
                    np.int32 if output_name in ("labels", "num_det") else self.dtype_np
                )
                output_value = ort.OrtValue.ortvalue_from_shape_and_type(
                    output_shape, output_dtype, "cuda", device_id
                )
                self.output_ortvalues[output_name] = output_value
                self.io_binding.bind_output(
                    name=output_name,
                    device_type="cuda",
                    device_id=device_id,
                    element_type=output_dtype,
                    shape=output_shape,
                    buffer_ptr=output_value.data_ptr(),
                )

    def warmup(self, warmup_runs: int) -> None:
        if warmup_runs <= 0:
            return
        dummy = np.zeros((1, 3, self.imgsz, self.imgsz), dtype=self.dtype_np)
        for _ in range(int(warmup_runs)):
            _ = self.run_from_numpy(dummy)

    def run(
        self, batch_images: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], float]:
        batch_np = batch_images.detach().cpu().numpy().astype(self.dtype_np, copy=False)
        return self.run_from_numpy(batch_np)

    def run_from_numpy(
        self, batch_np: np.ndarray
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], float]:
        if self.io_binding is not None and self.input_ortvalue is not None:
            self.input_ortvalue.update_inplace(batch_np)
            start = time.perf_counter()
            self.session.run_with_iobinding(self.io_binding)
            inf_ms = (time.perf_counter() - start) * 1e3
            outputs = self.io_binding.copy_outputs_to_cpu()
        else:
            start = time.perf_counter()
            outputs = self.session.run(self.output_names, {self.input_name: batch_np})
            inf_ms = (time.perf_counter() - start) * 1e3

        named = dict(zip(self.output_names, outputs))
        boxes = torch.from_numpy(np.asarray(named["boxes"]))
        scores = torch.from_numpy(np.asarray(named["scores"]))
        labels = torch.from_numpy(
            np.asarray(named["labels"]).astype(np.int32, copy=False)
        )
        num_det = torch.from_numpy(
            np.asarray(named["num_det"]).astype(np.int32, copy=False)
        )
        return (boxes, scores, labels, num_det), inf_ms


class TensorRTRunner(object):
    def __init__(self, engine_path: Path, device: torch.device, imgsz: int) -> None:
        if device.type != "cuda":
            raise RuntimeError("TensorRT benchmark requires a CUDA device.")
        try:
            import tensorrt as trt
        except Exception as exc:
            raise RuntimeError("Failed to import tensorrt: %s" % exc)

        self.trt = trt
        self.engine_path = engine_path
        self.device = device
        self.imgsz = int(imgsz)
        self.logger = trt.Logger(trt.Logger.ERROR)  # type: ignore
        self.runtime = trt.Runtime(self.logger)  # type: ignore
        with open(str(engine_path), "rb") as handle:
            engine_bytes = handle.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(
                "Failed to deserialize TensorRT engine: %s" % engine_path
            )
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        self.output_tensors = {}  # type: Dict[str, torch.Tensor]
        self.input_tensor = None  # type: Optional[torch.Tensor]
        self.binding_addresses = None  # type: Optional[List[int]]
        self.use_v3_api = hasattr(self.engine, "num_io_tensors")
        self._prepare_bindings()

    def _prepare_bindings(self) -> None:
        trt = self.trt
        if self.use_v3_api:
            tensor_names = [
                self.engine.get_tensor_name(i)
                for i in range(self.engine.num_io_tensors)
            ]
            self.input_name = None  # type: Optional[str]
            self.output_names = []  # type: List[str]
            for name in tensor_names:
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:  # type: ignore
                    self.input_name = name
                else:
                    self.output_names.append(name)
            if self.input_name is None:
                raise RuntimeError("TensorRT engine has no input tensor.")
            try:
                self.context.set_input_shape(
                    self.input_name, (1, 3, self.imgsz, self.imgsz)
                )
            except Exception:
                pass
            input_shape = tuple(
                int(x) for x in self.context.get_tensor_shape(self.input_name)
            )
            input_dtype = trt_dtype_to_torch(
                self.engine.get_tensor_dtype(self.input_name), trt
            )
            self.input_tensor = torch.empty(
                input_shape, device=self.device, dtype=input_dtype
            )
            self.context.set_tensor_address(
                self.input_name, int(self.input_tensor.data_ptr())
            )
            for name in self.output_names:
                output_shape = tuple(
                    int(x) for x in self.context.get_tensor_shape(name)
                )
                output_dtype = trt_dtype_to_torch(
                    self.engine.get_tensor_dtype(name), trt
                )
                output_tensor = torch.empty(
                    output_shape, device=self.device, dtype=output_dtype
                )
                self.output_tensors[name] = output_tensor
                self.context.set_tensor_address(name, int(output_tensor.data_ptr()))
        else:
            self.input_index = None  # type: Optional[int]
            self.output_names = []  # type: List[str]
            self.binding_addresses = [0 for _ in range(int(self.engine.num_bindings))]
            for binding_index in range(int(self.engine.num_bindings)):
                name = self.engine.get_binding_name(binding_index)
                if self.engine.binding_is_input(binding_index):
                    self.input_index = binding_index
                else:
                    self.output_names.append(name)
            if self.input_index is None:
                raise RuntimeError("TensorRT engine has no input binding.")
            try:
                self.context.set_binding_shape(
                    self.input_index, (1, 3, self.imgsz, self.imgsz)
                )
            except Exception:
                pass
            input_shape = tuple(
                int(x) for x in self.context.get_binding_shape(self.input_index)
            )
            input_dtype = trt_dtype_to_torch(
                self.engine.get_binding_dtype(self.input_index), trt
            )
            self.input_tensor = torch.empty(
                input_shape, device=self.device, dtype=input_dtype
            )
            self.binding_addresses[self.input_index] = int(self.input_tensor.data_ptr())
            for binding_index in range(int(self.engine.num_bindings)):
                if self.engine.binding_is_input(binding_index):
                    continue
                name = self.engine.get_binding_name(binding_index)
                output_shape = tuple(
                    int(x) for x in self.context.get_binding_shape(binding_index)
                )
                output_dtype = trt_dtype_to_torch(
                    self.engine.get_binding_dtype(binding_index), trt
                )
                output_tensor = torch.empty(
                    output_shape, device=self.device, dtype=output_dtype
                )
                self.output_tensors[name] = output_tensor
                self.binding_addresses[binding_index] = int(output_tensor.data_ptr())

    def warmup(self, warmup_runs: int) -> None:
        if warmup_runs <= 0:
            return
        assert self.input_tensor is not None
        dummy = torch.zeros_like(self.input_tensor)
        for _ in range(int(warmup_runs)):
            self.input_tensor.copy_(dummy)
            self._execute()
            sync_if_cuda(self.device)

    def _execute(self) -> None:
        stream = torch.cuda.current_stream(self.device)
        stream_handle = stream.cuda_stream
        if self.use_v3_api:
            ok = self.context.execute_async_v3(stream_handle)
        else:
            assert self.binding_addresses is not None
            ok = self.context.execute_async_v2(self.binding_addresses, stream_handle)
        if not ok:
            raise RuntimeError("TensorRT execution failed.")

    def run(
        self, batch_images: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], float]:
        assert self.input_tensor is not None
        input_dtype = self.input_tensor.dtype
        batch_gpu = batch_images.to(self.device, dtype=input_dtype, non_blocking=True)
        self.input_tensor.copy_(batch_gpu)
        sync_if_cuda(self.device)
        start = time.perf_counter()
        self._execute()
        sync_if_cuda(self.device)
        inf_ms = (time.perf_counter() - start) * 1e3

        boxes = self.output_tensors["boxes"].detach().cpu().clone()
        scores = self.output_tensors["scores"].detach().cpu().clone()
        labels = (
            self.output_tensors["labels"].detach().cpu().clone().to(dtype=torch.int32)
        )
        num_det = (
            self.output_tensors["num_det"].detach().cpu().clone().to(dtype=torch.int32)
        )
        return (boxes, scores, labels, num_det), inf_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark torchvision detection models with a fixed-output export path comparable to Ultralytics benchmark mode."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/coco.yaml"),
        help="Ultralytics-style dataset YAML (default: configs/coco.yaml)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        choices=sorted(MODEL_ZOO.keys()),
        help="Torchvision models to benchmark.",
    )
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=list(DEFAULT_SIZES),
        help="One or more square benchmark sizes.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(DEFAULT_FORMATS),
        choices=sorted(FORMAT_TO_NAME.keys()),
        help="Benchmark formats to run.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Number of independent benchmark repeats per model/size/format/precision.",
    )
    parser.add_argument(
        "--include-half",
        dest="include_half",
        action="store_true",
        default=True,
        help="Include FP16 rows in addition to FP32 rows (default: enabled).",
    )
    parser.add_argument(
        "--no-include-half",
        dest="include_half",
        action="store_false",
        help="Disable FP16 rows.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Benchmark batch size. This script keeps batch=1 for parity with Ultralytics benchmark mode.",
    )
    parser.add_argument("--device", default="0", help='0, "cuda:0", or "cpu"')
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--project",
        type=str,
        default="runs/val_bench",
        help="Output root folder.",
    )
    parser.add_argument("--out-csv", type=Path, default=Path("val_jetson_test.csv"))
    parser.add_argument(
        "--exist-ok",
        dest="exist_ok",
        action="store_true",
        default=True,
        help="Reuse existing run folders (default: enabled).",
    )
    parser.add_argument(
        "--no-exist-ok",
        dest="exist_ok",
        action="store_false",
        help="Fail if a run folder already exists.",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--score-thr", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument(
        "--limit", type=int, default=0, help="Debug: evaluate only the first N images."
    )
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=0,
        help="ONNX opset to export with. 0 means auto-select a best-effort value.",
    )
    return parser.parse_args()


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "-":
        return None
    try:
        return float(text)
    except Exception:
        return None


def printable(value: Any) -> str:
    if value is None:
        return "nan"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return "%.4f" % value
    return str(value)


def ensure_csv_schema(out_csv: Path) -> None:
    out_csv = out_csv.expanduser().resolve()
    if not out_csv.exists():
        return
    try:
        with out_csv.open("r", newline="") as handle:
            first_line = handle.readline().strip("\n\r")
    except Exception:
        return
    current_header = first_line.split(",") if first_line else []
    if current_header == FIELDNAMES:
        return
    backup = out_csv.with_name(
        out_csv.stem + ".legacy_" + str(int(time.time())) + out_csv.suffix
    )
    out_csv.rename(backup)
    print(
        "[INFO] Existing CSV header differs from the new schema. Moved old file to: %s"
        % backup
    )


def write_csv(rows: Iterable[Dict[str, Any]], out_csv: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    out_csv = out_csv.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ensure_csv_schema(out_csv)
    new_file = not out_csv.exists()
    with out_csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()


def resolve_coco_from_ultralytics_yaml(data_yaml: Path) -> Tuple[Path, Path]:
    data_yaml = data_yaml.expanduser().resolve()
    with data_yaml.open("r") as handle:
        data = yaml.safe_load(handle)

    root = Path(str(data.get("path", ".")))
    if not root.is_absolute():
        candidates = [
            (data_yaml.parent / root).resolve(),
            (data_yaml.parent.parent / root).resolve(),
        ]
        chosen = None
        for candidate in candidates:
            if candidate.exists():
                chosen = candidate
                break
        root = chosen if chosen is not None else candidates[0]

    val = data.get("val")
    if val is None:
        raise KeyError("%s missing required key: val" % data_yaml)

    val_path = Path(str(val))
    if not val_path.is_absolute():
        val_path = (root / val_path).resolve()

    anno_json = root / "annotations" / "instances_val2017.json"
    if not anno_json.is_file():
        ann_dir = root / "annotations"
        if ann_dir.is_dir():
            candidates = sorted(ann_dir.glob("instances_val*.json"))
            if candidates:
                anno_json = candidates[0]
        if not anno_json.is_file():
            raise FileNotFoundError(
                "Could not find instances_val*.json under %s" % ann_dir
            )

    if val_path.is_file() and val_path.suffix == ".txt":
        with val_path.open("r") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        if not lines:
            raise ValueError("Empty val txt file: %s" % val_path)
        img_dir = root / Path(lines[0])
        img_dir = img_dir.parent
    else:
        img_dir = val_path

    if not img_dir.is_absolute():
        img_dir = (root / img_dir).resolve()

    if not img_dir.is_dir():
        raise FileNotFoundError("Validation images directory not found: %s" % img_dir)

    return img_dir, anno_json


def letterbox_pil_to_tensor(
    image: Image.Image, imgsz: int, color: int = 114
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    orig_w, orig_h = image.size
    scale = min(float(imgsz) / float(orig_w), float(imgsz) / float(orig_h))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    resized = image.resize((new_w, new_h), PIL_BILINEAR)
    canvas = Image.new("RGB", (int(imgsz), int(imgsz)), (color, color, color))
    pad_x = int((imgsz - new_w) // 2)
    pad_y = int((imgsz - new_h) // 2)
    canvas.paste(resized, (pad_x, pad_y))

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    meta = {
        "orig_w": int(orig_w),
        "orig_h": int(orig_h),
        "scale": float(scale),
        "pad_x": int(pad_x),
        "pad_y": int(pad_y),
        "new_w": int(new_w),
        "new_h": int(new_h),
        "imgsz": int(imgsz),
    }
    return tensor, meta


def pick_device(device: Any) -> torch.device:
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    if isinstance(device, int):
        device = "cuda:%d" % device
    device_str = str(device)
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def cuda_device_index(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def weights_descriptor(weights: Any) -> str:
    name = getattr(weights, "name", "DEFAULT")
    return "torchvision::{cls}.{name}".format(cls=weights.__class__.__name__, name=name)


def file_size_mb(path_str: str) -> Optional[float]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        return None
    try:
        return float(path.stat().st_size) / (1024.0 * 1024.0)
    except Exception:
        return None


def trt_dtype_to_torch(dtype: Any, trt_module: Any) -> torch.dtype:
    mapping = {
        trt_module.float32: torch.float32,
        trt_module.float16: torch.float16,
        trt_module.int32: torch.int32,
        trt_module.int8: torch.int8,
        trt_module.bool: torch.bool,
    }
    if dtype not in mapping:
        raise RuntimeError("Unsupported TensorRT dtype: %s" % dtype)
    return mapping[dtype]


def find_trtexec() -> Optional[str]:
    candidate = shutil.which("trtexec")
    if candidate:
        return candidate
    common_paths = [
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/tensorrt/bin/trtexec",
        "/usr/lib/aarch64-linux-gnu/tensorrt/bin/trtexec",
    ]
    for path in common_paths:
        if Path(path).exists():
            return path
    return None


def infer_best_onnx_opset(cuda_enabled: bool) -> int:
    version_text = torch.__version__.split("+", 1)[0]
    parts = version_text.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1])
    except Exception:
        major, minor = (2, 0)

    if (major, minor) >= (2, 4):
        try:
            from torch.onnx import utils as onnx_utils  # type: ignore

            opset = int(onnx_utils._constants.ONNX_MAX_OPSET) - 1  # type: ignore[attr-defined]
        except Exception:
            opset = 20
        if (major, minor) >= (2, 9):
            opset = min(opset, 20)
        if cuda_enabled:
            opset -= 2
    else:
        mapping = {
            (1, 8): 12,
            (1, 9): 12,
            (1, 10): 13,
            (1, 11): 14,
            (1, 12): 15,
            (1, 13): 17,
            (2, 0): 17,
            (2, 1): 17,
            (2, 2): 17,
            (2, 3): 17,
            (2, 4): 20,
            (2, 5): 20,
            (2, 6): 20,
            (2, 7): 20,
            (2, 8): 23,
        }
        opset = mapping.get((major, minor), 17)

    try:
        import onnx

        opset = min(int(opset), int(onnx.defs.onnx_opset_version()))
    except Exception:
        opset = int(opset)
    return max(11, int(opset))


def build_model(
    model_name: str,
    device: torch.device,
    score_thr: float,
    max_det: int,
    imgsz: int,
    half: bool,
) -> Tuple[nn.Module, Any]:
    if model_name not in MODEL_ZOO:
        raise KeyError("Unknown model: %s" % model_name)

    ctor, weights_enum = MODEL_ZOO[model_name]
    weights = weights_enum.DEFAULT
    model = ctor(weights=weights).eval().to(device)

    if model_name in ("fasterrcnn_resnet50_fpn", "maskrcnn_resnet50_fpn"):
        model.roi_heads.score_thresh = float(score_thr)
        model.roi_heads.detections_per_img = int(max_det)
    elif model_name == "retinanet_resnet50_fpn":
        model.score_thresh = float(score_thr)
        model.detections_per_img = int(max_det)

    transform = getattr(model, "transform", None)
    if transform is not None:
        if hasattr(transform, "min_size"):
            transform.min_size = (int(imgsz),)
        if hasattr(transform, "max_size"):
            transform.max_size = int(imgsz)
        if hasattr(transform, "fixed_size"):
            transform.fixed_size = None
        if hasattr(transform, "size_divisible"):
            transform.size_divisible = 32

    if half:
        model = model.half()
    return model, weights


def build_label_to_coco_catid(weights: Any, coco: COCO) -> Dict[int, int]:
    categories = list(weights.meta["categories"])
    coco_cats = coco.loadCats(coco.getCatIds())
    name_to_coco_id = dict((cat["name"], int(cat["id"])) for cat in coco_cats)
    mapping = {}  # type: Dict[int, int]
    for label_index, name in enumerate(categories):
        if name == "__background__":
            continue
        coco_id = name_to_coco_id.get(name)
        if coco_id is not None:
            mapping[int(label_index)] = int(coco_id)
    return mapping


def reverse_letterbox(
    box: Sequence[float], meta: Dict[str, Any]
) -> Optional[List[float]]:
    scale = float(meta["scale"])
    pad_x = float(meta["pad_x"])
    pad_y = float(meta["pad_y"])
    orig_w = float(meta["orig_w"])
    orig_h = float(meta["orig_h"])
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale
    x1 = max(0.0, min(orig_w, x1))
    y1 = max(0.0, min(orig_h, y1))
    x2 = max(0.0, min(orig_w, x2))
    y2 = max(0.0, min(orig_h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def fixed_outputs_to_coco_json(
    outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    image_ids: Sequence[int],
    metas: Sequence[Dict[str, Any]],
    label_to_catid: Dict[int, int],
    score_thr: float,
    max_det: int,
) -> List[Dict[str, Any]]:
    boxes, scores, labels, num_det = outputs
    boxes = boxes.to(dtype=torch.float32)
    scores = scores.to(dtype=torch.float32)
    labels = labels.to(dtype=torch.int32)
    num_det = num_det.to(dtype=torch.int32)

    results = []  # type: List[Dict[str, Any]]
    batch_size = len(image_ids)
    for batch_index in range(batch_size):
        n = min(int(num_det[batch_index].item()), int(max_det))
        if n <= 0:
            continue
        for det_index in range(n):
            score = float(scores[batch_index, det_index].item())
            if score < float(score_thr):
                continue
            label = int(labels[batch_index, det_index].item())
            coco_id = label_to_catid.get(label)
            if coco_id is None:
                continue
            box = boxes[batch_index, det_index].tolist()
            restored = reverse_letterbox(box, metas[batch_index])
            if restored is None:
                continue
            x1, y1, x2, y2 = restored
            results.append(
                {
                    "image_id": int(image_ids[batch_index]),
                    "category_id": int(coco_id),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": score,
                }
            )
    return results


@torch.no_grad()
def warmup_torch_module(
    module: Any, device: torch.device, imgsz: int, half: bool, warmup_runs: int
) -> None:
    if warmup_runs <= 0:
        return
    dtype = torch.float16 if half else torch.float32
    dummy = torch.zeros((1, 3, imgsz, imgsz), device=device, dtype=dtype)
    for _ in range(int(warmup_runs)):
        _ = module(dummy)
    sync_if_cuda(device)


@torch.no_grad()
def run_torch_module(
    module: Any,
    batch_images: torch.Tensor,
    device: torch.device,
    half: bool,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], float]:
    dtype = torch.float16 if half else torch.float32
    images_device = batch_images.to(
        device, dtype=dtype, non_blocking=(device.type == "cuda")
    )
    sync_if_cuda(device)
    start = time.perf_counter()
    boxes, scores, labels, num_det = module(images_device)
    sync_if_cuda(device)
    inf_ms = (time.perf_counter() - start) * 1e3
    return (
        boxes.detach().cpu(),
        scores.detach().cpu(),
        labels.detach().cpu().to(dtype=torch.int32),
        num_det.detach().cpu().to(dtype=torch.int32),
    ), inf_ms


def export_torchscript(
    wrapper: nn.Module, dummy_input: torch.Tensor, out_path: Path, device: torch.device
) -> Any:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced = torch.jit.trace(wrapper, dummy_input, strict=False)
    traced = torch.jit.freeze(traced.eval())  # type: ignore
    traced.save(str(out_path))
    loaded = torch.jit.load(str(out_path), map_location=device)
    loaded = loaded.eval()
    try:
        loaded = torch.jit.freeze(loaded)
    except Exception:
        pass
    return loaded


def export_onnx(
    wrapper: nn.Module, dummy_input: torch.Tensor, out_path: Path, opset: int
) -> None:
    try:
        import onnx  # noqa: F401  # validate dependency presence before export
    except Exception as exc:
        raise RuntimeError("Failed to import onnx: %s" % exc)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy_input,  # type: ignore
        str(out_path),
        export_params=True,
        opset_version=int(opset),
        do_constant_folding=True,
        input_names=["images"],
        output_names=["boxes", "scores", "labels", "num_det"],
        dynamic_axes=None,
    )

    try:
        import onnx

        model = onnx.load(str(out_path))
        onnx.checker.check_model(model)
    except Exception as exc:
        raise RuntimeError("ONNX export validation failed: %s" % exc)


def build_tensorrt_engine(onnx_path: Path, engine_path: Path, half: bool) -> None:
    trtexec = find_trtexec()
    if not trtexec:
        raise RuntimeError(
            "trtexec was not found in PATH or common TensorRT install locations."
        )

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        trtexec,
        "--onnx=%s" % onnx_path,
        "--saveEngine=%s" % engine_path,
        "--skipInference",
    ]
    if half:
        cmd.append("--fp16")

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        raise RuntimeError(
            "trtexec failed with exit code %s\n%s"
            % (result.returncode, combined.strip())
        )
    if not engine_path.exists():
        raise RuntimeError("TensorRT engine was not created: %s" % engine_path)


def coco_map_via_ultralytics(
    dataloader: DataLoader,
    save_dir: Path,
    pred_json: Path,
    anno_json: Path,
    jdict: List[Dict[str, Any]],
) -> Dict[str, Any]:
    save_dir = Path(save_dir)
    pred_json = Path(pred_json)
    anno_json = Path(anno_json)

    if not pred_json.is_file():
        raise FileNotFoundError("pred_json not found: %s" % pred_json)
    if not anno_json.is_file():
        raise FileNotFoundError("anno_json not found: %s" % anno_json)
    if not jdict:
        raise ValueError("Prediction JSON is empty.")

    im_files = getattr(dataloader.dataset, "im_files", None)
    if not im_files:
        raise AttributeError(
            "dataloader.dataset must expose im_files for Ultralytics COCO evaluation."
        )

    eval_img_ids = []  # type: List[int]
    for path in im_files:
        eval_img_ids.append(int(Path(path).stem))
    eval_set = set(eval_img_ids)

    filtered = []  # type: List[Dict[str, Any]]
    for pred in jdict:
        try:
            image_id = int(pred.get("image_id", -1))
        except Exception:
            continue
        if image_id in eval_set:
            filtered.append(pred)

    if not filtered:
        raise ValueError(
            "Predictions became empty after filtering to evaluated COCO image IDs."
        )

    validator = DetectionValidator(
        dataloader=dataloader, save_dir=save_dir, args={"save_json": True}
    )
    validator.is_coco = True
    validator.is_lvis = False
    validator.jdict = filtered
    try:
        validator.args.save_json = True
    except Exception:
        pass
    stats = {}  # type: Dict[str, Any]
    stats = validator.coco_evaluate(stats, pred_json=pred_json, anno_json=anno_json)  # type: ignore
    return stats


def build_error_row(
    spec: RunSpec, args: argparse.Namespace, error: str
) -> Dict[str, Any]:
    precision = "fp16" if spec.half else "fp32"
    format_name = FORMAT_TO_NAME[spec.format_key]
    run_name = "torchvision_{model}_img{imgsz}_{fmt}_{prec}_r{repeat}".format(
        model=spec.model,
        imgsz=spec.imgsz,
        fmt=spec.format_key,
        prec=precision,
        repeat=spec.repeat,
    )
    save_dir = str(Path(args.project).expanduser().resolve() / run_name)
    return {
        "framework": "torchvision",
        "model": spec.model,
        "weights": "",
        "imgsz": spec.imgsz,
        "batch": args.batch,
        "device": str(args.device),
        "repeat": spec.repeat,
        "half": int(spec.half),
        "precision": precision,
        "format_name": format_name,
        "format_arg": FORMAT_TO_ARG[spec.format_key],
        "backend": format_name.lower(),
        "runtime_provider": "",
        "benchmark_impl": "torchvision_custom_benchmark_v2",
        "input_geometry": "external_letterbox_square",
        "resize_mode": "letterbox",
        "conf": float(args.score_thr),
        "max_det": int(args.max_det),
        "speed_preprocess_ms": "",
        "speed_inference_ms": "",
        "speed_postprocess_ms": "",
        "ap50_95": "",
        "fps": "",
        "status": "error",
        "error": error,
        "artifact_path": "",
        "artifact_size_mb": "",
        "save_dir": save_dir,
        "run_name": run_name,
        "metric_key": METRIC_KEY,
        "benchmark_wall_time_s": "",
    }


def run_one(
    spec: RunSpec, img_dir: Path, anno_json: Path, args: argparse.Namespace
) -> Dict[str, Any]:
    if int(args.batch) != 1:
        return build_error_row(
            spec, args, "This script keeps batch=1 to match Ultralytics benchmark mode."
        )

    device = pick_device(args.device)
    if spec.half and device.type != "cuda":
        return build_error_row(
            spec,
            args,
            "FP16 benchmarking is only enabled for CUDA devices in this script.",
        )
    if spec.format_key == "trt" and device.type != "cuda":
        return build_error_row(
            spec, args, "TensorRT benchmarking requires a CUDA device."
        )

    precision = "fp16" if spec.half else "fp32"
    run_name = "torchvision_{model}_img{imgsz}_{fmt}_{prec}_r{repeat}".format(
        model=spec.model,
        imgsz=spec.imgsz,
        fmt=spec.format_key,
        prec=precision,
        repeat=spec.repeat,
    )
    save_dir = Path(args.project).expanduser().resolve() / run_name
    if save_dir.exists() and not bool(args.exist_ok):
        return build_error_row(
            spec,
            args,
            "Run directory already exists and --no-exist-ok was used: %s" % save_dir,
        )
    save_dir.mkdir(parents=True, exist_ok=True)

    wall_start = time.perf_counter()

    try:
        model, weights = build_model(
            spec.model,
            device=device,
            score_thr=float(args.score_thr),
            max_det=int(args.max_det),
            imgsz=int(spec.imgsz),
            half=bool(spec.half),
        )
    except Exception as exc:
        return build_error_row(
            spec, args, "Failed to build torchvision model: %s" % exc
        )

    try:
        dataset = CocoLetterboxDataset(
            img_dir=img_dir, anno_json=anno_json, imgsz=int(spec.imgsz)
        )
        if int(args.limit) > 0:
            dataset.img_ids = dataset.img_ids[: int(args.limit)]
            dataset.im_files = dataset.im_files[: int(args.limit)]
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=int(args.workers),
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_fn,
        )
    except Exception as exc:
        return build_error_row(spec, args, "Failed to build dataloader: %s" % exc)

    try:
        label_to_catid = build_label_to_coco_catid(weights, dataset.coco)
    except Exception as exc:
        return build_error_row(spec, args, "Failed to build label mapping: %s" % exc)

    wrapper = TVDetExportWrapper(model, int(args.max_det)).eval().to(device)
    dummy_dtype = torch.float16 if spec.half else torch.float32
    dummy_input = torch.zeros(
        (1, 3, int(spec.imgsz), int(spec.imgsz)), device=device, dtype=dummy_dtype
    )

    artifact_path = ""
    runtime_provider = str(device)
    backend = "pytorch_eager"
    onnx_opset = (
        int(args.onnx_opset)
        if int(args.onnx_opset) > 0
        else infer_best_onnx_opset(device.type == "cuda")
    )

    try:
        if spec.format_key == "pytorch":
            warmup_torch_module(
                wrapper, device, int(spec.imgsz), bool(spec.half), int(args.warmup)
            )

            def infer(
                batch_images: torch.Tensor,
            ) -> Tuple[
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], float
            ]:
                return run_torch_module(wrapper, batch_images, device, bool(spec.half))

            backend = "pytorch_eager"
            runtime_provider = str(device)

        elif spec.format_key == "torchscript":
            artifact = save_dir / (spec.model + "_img%d.torchscript" % int(spec.imgsz))
            script_module = export_torchscript(wrapper, dummy_input, artifact, device)
            artifact_path = str(artifact)
            warmup_torch_module(
                script_module,
                device,
                int(spec.imgsz),
                bool(spec.half),
                int(args.warmup),
            )

            def infer(
                batch_images: torch.Tensor,
            ) -> Tuple[
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], float
            ]:
                return run_torch_module(
                    script_module, batch_images, device, bool(spec.half)
                )

            backend = "torchscript"
            runtime_provider = str(device)

        elif spec.format_key == "onnx":
            artifact = save_dir / (spec.model + "_img%d.onnx" % int(spec.imgsz))
            export_onnx(wrapper, dummy_input, artifact, onnx_opset)
            artifact_path = str(artifact)
            onnx_runner = OnnxRuntimeRunner(
                artifact, device, int(spec.imgsz), int(args.max_det), bool(spec.half)
            )
            runtime_provider = onnx_runner.provider
            if spec.half and runtime_provider != "CUDAExecutionProvider":
                return build_error_row(
                    spec,
                    args,
                    "FP16 ONNX benchmark requires ONNX Runtime CUDAExecutionProvider; got '%s'."
                    % runtime_provider,
                )
            onnx_runner.warmup(int(args.warmup))

            def infer(
                batch_images: torch.Tensor,
            ) -> Tuple[
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], float
            ]:
                return onnx_runner.run(batch_images)

            backend = "onnxruntime"

        elif spec.format_key == "trt":
            onnx_artifact = save_dir / (spec.model + "_img%d.onnx" % int(spec.imgsz))
            engine_artifact = save_dir / (
                spec.model + "_img%d.engine" % int(spec.imgsz)
            )
            export_onnx(wrapper, dummy_input, onnx_artifact, onnx_opset)
            build_tensorrt_engine(onnx_artifact, engine_artifact, bool(spec.half))
            artifact_path = str(engine_artifact)
            trt_runner = TensorRTRunner(engine_artifact, device, int(spec.imgsz))
            trt_runner.warmup(int(args.warmup))

            def infer(
                batch_images: torch.Tensor,
            ) -> Tuple[
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], float
            ]:
                return trt_runner.run(batch_images)

            backend = "tensorrt"
            runtime_provider = "tensorrt"

        else:
            return build_error_row(
                spec, args, "Unsupported format key: %s" % spec.format_key
            )

    except Exception as exc:
        return build_error_row(spec, args, "Backend setup/export failed: %s" % exc)

    all_predictions = []  # type: List[Dict[str, Any]]
    inference_ms_total = 0.0
    num_images = 0

    progress = tqdm(dataloader, desc=run_name, unit="img")
    try:
        for image_ids, batch_images, metas in progress:
            outputs, inf_ms = infer(batch_images)
            num_images += len(image_ids)
            inference_ms_total += float(inf_ms)
            all_predictions.extend(
                fixed_outputs_to_coco_json(
                    outputs=outputs,
                    image_ids=image_ids,
                    metas=metas,
                    label_to_catid=label_to_catid,
                    score_thr=float(args.score_thr),
                    max_det=int(args.max_det),
                )
            )
            progress.set_postfix(
                inf="%.2fms" % (inference_ms_total / float(max(1, num_images)))
            )
    except Exception as exc:
        return build_error_row(spec, args, "Inference loop failed: %s" % exc)
    finally:
        progress.close()

    pred_json = save_dir / "predictions.json"
    try:
        with pred_json.open("w") as handle:
            json.dump(all_predictions, handle)
    except Exception as exc:
        return build_error_row(spec, args, "Failed to write predictions JSON: %s" % exc)

    try:
        stats = coco_map_via_ultralytics(
            dataloader=dataloader,
            save_dir=save_dir,
            pred_json=pred_json,
            anno_json=anno_json,
            jdict=all_predictions,
        )
        ap50_95 = float(stats.get(METRIC_KEY, float("nan")))
        status = "success"
        error = ""
    except Exception as exc:
        ap50_95 = float("nan")
        status = "error"
        error = "COCO evaluation failed: %s" % exc

    wall_time_s = time.perf_counter() - wall_start
    mean_inf_ms = inference_ms_total / float(max(1, num_images))
    fps = None if mean_inf_ms <= 0 else (1000.0 / mean_inf_ms)

    row = {
        "framework": "torchvision",
        "model": spec.model,
        "weights": weights_descriptor(weights),
        "imgsz": int(spec.imgsz),
        "batch": 1,
        "device": str(args.device),
        "repeat": int(spec.repeat),
        "half": int(spec.half),
        "precision": precision,
        "format_name": FORMAT_TO_NAME[spec.format_key],
        "format_arg": FORMAT_TO_ARG[spec.format_key],
        "backend": backend,
        "runtime_provider": runtime_provider,
        "benchmark_impl": "torchvision_custom_benchmark_v2",
        "input_geometry": "external_letterbox_square",
        "resize_mode": "letterbox",
        "conf": float(args.score_thr),
        "max_det": int(args.max_det),
        "speed_preprocess_ms": "",
        "speed_inference_ms": mean_inf_ms,
        "speed_postprocess_ms": "",
        "ap50_95": ap50_95,
        "fps": fps,
        "status": status,
        "error": error,
        "artifact_path": artifact_path,
        "artifact_size_mb": file_size_mb(artifact_path),
        "save_dir": str(save_dir),
        "run_name": run_name,
        "metric_key": METRIC_KEY,
        "benchmark_wall_time_s": wall_time_s,
    }
    return row


def main() -> None:
    args = parse_args()
    img_dir, anno_json = resolve_coco_from_ultralytics_yaml(args.data)

    precisions = [False]
    if args.include_half:
        precisions.append(True)

    rows_written = 0
    for model_name in args.models:
        for imgsz in args.imgsz:
            for format_key in args.formats:
                for half in precisions:
                    for repeat in range(1, int(args.repeats) + 1):
                        spec = RunSpec(
                            model=str(model_name),
                            imgsz=int(imgsz),
                            format_key=str(format_key),
                            half=bool(half),
                            repeat=int(repeat),
                        )
                        row = run_one(spec, img_dir, anno_json, args)
                        print(
                            "[{framework}] {model:>28} imgsz={imgsz:>4} fmt={fmt:>11} prec={prec:>4} run={run:>2} | "
                            "status={status:>8} inf={inf}ms AP50-95={ap}".format(
                                framework=row["framework"],
                                model=row["model"],
                                imgsz=row["imgsz"],
                                fmt=row["format_name"],
                                prec=row["precision"],
                                run=row["repeat"],
                                status=row["status"],
                                inf=printable(row.get("speed_inference_ms")),
                                ap=printable(row.get("ap50_95")),
                            )
                        )
                        if row.get("error"):
                            print("  -> %s" % row["error"])
                        write_csv([row], args.out_csv)
                        rows_written += 1

    print(
        "\nSaved %d rows to: %s" % (rows_written, args.out_csv.expanduser().resolve())
    )


if __name__ == "__main__":
    main()
