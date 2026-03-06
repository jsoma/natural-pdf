"""Generic YOLO-format ONNX checkbox detector.

Runs Ultralytics YOLOv8/YOLO11 ONNX exports. Only depends on
onnxruntime and numpy (no torch, no cv2). All heavy imports are
lazy (inside detect(), never at module import time).
"""

import importlib.util
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .base import CheckboxDetector, DetectionContext
from .checkbox_options import BaseCheckboxOptions, OnnxCheckboxOptions

logger = logging.getLogger(__name__)


class OnnxCheckboxDetector(CheckboxDetector):
    """Runs Ultralytics YOLO-format ONNX models for checkbox detection."""

    def __init__(self):
        super().__init__()
        self._session_cache: Dict[str, Any] = {}

    def is_available(self) -> bool:
        return importlib.util.find_spec("onnxruntime") is not None

    def detect(
        self,
        image: Image.Image,
        options: BaseCheckboxOptions,
        context: Optional[DetectionContext] = None,
    ) -> List[Dict[str, Any]]:
        import onnxruntime as ort  # Lazy import

        if not isinstance(options, OnnxCheckboxOptions):
            options = OnnxCheckboxOptions(
                confidence=options.confidence,
                resolution=options.resolution,
                device=options.device,
            )

        model_path = self._resolve_model_path(options)
        if model_path is None:
            self.logger.error("No ONNX model path specified")
            return []

        session = self._get_session(model_path)
        input_shape = session.get_inputs()[0].shape  # e.g. [1, 3, 640, 640]

        # Use model's expected input size if available
        model_size = options.input_size
        if isinstance(input_shape, list) and len(input_shape) == 4:
            if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
                model_size = input_shape[2]

        # Auto-read class names from model metadata if not provided
        if not options.class_names:
            meta_names = self._read_class_names(session)
            if meta_names:
                import dataclasses as _dc

                options = _dc.replace(options, class_names=meta_names)

        # Decide whether to use SAHI tiling
        img_w, img_h = image.size
        use_sahi = (
            options.sahi_enabled and max(img_w, img_h) > model_size * options.sahi_min_image_ratio
        )

        if use_sahi:
            return self._detect_sahi(image, session, model_size, options)
        else:
            return self._detect_single(image, session, model_size, options)

    def _detect_single(
        self,
        image: Image.Image,
        session: Any,
        model_size: int,
        options: "OnnxCheckboxOptions",
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> List[Dict[str, Any]]:
        """Run inference on a single image/tile. Offset maps detections back to full image coords."""
        input_name = session.get_inputs()[0].name
        input_tensor, pad_info = self._preprocess(image, model_size)
        outputs = session.run(None, {input_name: input_tensor})
        detections = self._postprocess(outputs, image.size, pad_info, options)

        # Shift detections by tile offset
        if offset_x != 0 or offset_y != 0:
            for det in detections:
                bx0, by0, bx1, by1 = det["bbox"]
                det["bbox"] = (bx0 + offset_x, by0 + offset_y, bx1 + offset_x, by1 + offset_y)

        return detections

    def _detect_sahi(
        self,
        image: Image.Image,
        session: Any,
        model_size: int,
        options: "OnnxCheckboxOptions",
    ) -> List[Dict[str, Any]]:
        """SAHI: Slicing Aided Hyper Inference for small object detection.

        Tiles the image into overlapping crops at model_size, runs inference
        on each tile, maps detections back to full-image coordinates, and
        applies cross-tile NMS to remove duplicates from overlap regions.
        """
        img_w, img_h = image.size
        tile_size = model_size
        overlap = int(tile_size * options.sahi_overlap)
        stride = tile_size - overlap

        # Generate tile grid
        tiles = []
        y_positions = list(range(0, img_h, stride))
        x_positions = list(range(0, img_w, stride))
        for y in y_positions:
            for x in x_positions:
                tx1 = min(x + tile_size, img_w)
                ty1 = min(y + tile_size, img_h)
                tx0 = max(0, tx1 - tile_size)
                ty0 = max(0, ty1 - tile_size)
                tiles.append((tx0, ty0, tx1, ty1))

        self.logger.debug(
            "SAHI: %dx%d image → %d tiles (%dpx, %d%% overlap)",
            img_w,
            img_h,
            len(tiles),
            tile_size,
            int(options.sahi_overlap * 100),
        )

        # Run inference on each tile
        all_detections = []
        for tx0, ty0, tx1, ty1 in tiles:
            tile_img = image.crop((tx0, ty0, tx1, ty1))
            tile_dets = self._detect_single(
                tile_img,
                session,
                model_size,
                options,
                offset_x=tx0,
                offset_y=ty0,
            )
            all_detections.extend(tile_dets)

        if not all_detections:
            return []

        # Cross-tile NMS to remove duplicate detections from overlap regions
        boxes = np.array([d["bbox"] for d in all_detections])
        scores = np.array([d["confidence"] for d in all_detections])
        keep = self.nms(boxes, scores, options.nms_threshold)

        return [all_detections[i] for i in keep]

    def _resolve_model_path(self, options: OnnxCheckboxOptions) -> Optional[str]:
        """Resolve model path from options (local file or HuggingFace download)."""
        if options.model_path:
            return options.model_path

        if options.model_repo:
            try:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(
                    repo_id=options.model_repo,
                    filename=options.model_file or "model.onnx",
                    revision=options.model_revision,
                    token=False,
                )
                return path
            except ImportError:
                self.logger.error(
                    "huggingface_hub required to download models. "
                    "Install with: pip install huggingface_hub"
                )
                return None
            except Exception as e:
                self.logger.error("Failed to download ONNX model: %s", e)
                return None

        return None

    def _get_session(self, model_path: str) -> Any:
        """Get or create an ONNX runtime session (cached per path)."""
        if model_path not in self._session_cache:
            import onnxruntime as ort

            self._session_cache[model_path] = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
        return self._session_cache[model_path]

    def _read_class_names(self, session: Any) -> Optional[List[str]]:
        """Read class names from Ultralytics ONNX metadata if available."""
        try:
            meta = session.get_modelmeta()
            names_str = meta.custom_metadata_map.get("names")
            if names_str:
                import ast

                names_dict = ast.literal_eval(names_str)
                if isinstance(names_dict, dict):
                    return [names_dict[i] for i in sorted(names_dict.keys())]
        except Exception:
            pass
        return None

    def _preprocess(
        self,
        image: Image.Image,
        input_size: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Letterbox resize using PIL (no cv2). Matches Ultralytics reference.

        Returns:
            (input_tensor, pad_info) where pad_info has:
            - pad_left, pad_top: padding offsets
            - scale: resize scale factor
        """
        # Convert to greyscale (model trained on greyscale data)
        image = image.convert("L").convert("RGB")

        orig_w, orig_h = image.size

        # Calculate scale to fit within input_size
        scale = min(input_size / orig_w, input_size / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        # Resize image
        resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Create padded canvas at full input_size (model expects fixed dimensions)
        canvas = Image.new("RGB", (input_size, input_size), (114, 114, 114))
        pad_left = (input_size - new_w) // 2
        pad_top = (input_size - new_h) // 2
        canvas.paste(resized, (pad_left, pad_top))

        # Convert to numpy: HWC -> CHW, float32, 0-1
        arr = np.array(canvas, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # CHW
        arr = np.expand_dims(arr, axis=0)  # NCHW

        pad_info = {
            "pad_left": pad_left,
            "pad_top": pad_top,
            "scale": scale,
        }

        return arr, pad_info

    def _postprocess(
        self,
        outputs: List[np.ndarray],
        orig_size: Tuple[int, int],
        pad_info: Dict[str, float],
        options: OnnxCheckboxOptions,
    ) -> List[Dict[str, Any]]:
        """Decode Ultralytics YOLO format output.

        Expected output shape: (1, 4+num_classes, N) — xywh center + class scores.
        """
        raw = outputs[0]  # shape: (1, 4+num_classes, N)

        if raw.ndim != 3:
            self.logger.error("Unexpected ONNX output shape: %s", raw.shape)
            return []

        # Transpose to (N, 4+num_classes)
        preds = raw[0].T  # (N, 4+C)
        num_classes = preds.shape[1] - 4

        if num_classes <= 0:
            self.logger.error("Invalid number of classes in output: %d", num_classes)
            return []

        # Split boxes and scores
        cx = preds[:, 0]
        cy = preds[:, 1]
        w = preds[:, 2]
        h = preds[:, 3]
        class_scores = preds[:, 4:]  # (N, C)

        # Get max class score and index per detection
        max_scores = class_scores.max(axis=1)
        max_indices = class_scores.argmax(axis=1)

        # Determine which classes are checkboxes
        checkbox_indices = set(options.checkbox_class_indices or range(num_classes))

        # Filter by confidence and class
        mask = max_scores >= options.confidence
        if options.checkbox_class_indices is not None:
            class_mask = np.isin(max_indices, list(checkbox_indices))
            mask = mask & class_mask

        if not mask.any():
            return []

        # Convert xywh center to xyxy
        x0 = cx[mask] - w[mask] / 2
        y0 = cy[mask] - h[mask] / 2
        x1 = cx[mask] + w[mask] / 2
        y1 = cy[mask] + h[mask] / 2
        scores = max_scores[mask]
        class_ids = max_indices[mask]

        # Undo letterbox: orig_x = (model_x - pad_left) / scale
        pad_left = pad_info["pad_left"]
        pad_top = pad_info["pad_top"]
        scale = pad_info["scale"]

        x0 = (x0 - pad_left) / scale
        y0 = (y0 - pad_top) / scale
        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale

        # Clip to image bounds
        orig_w, orig_h = orig_size
        x0 = np.clip(x0, 0, orig_w)
        y0 = np.clip(y0, 0, orig_h)
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)

        # Apply NMS
        boxes = np.stack([x0, y0, x1, y1], axis=1)
        keep = self.nms(boxes, scores, options.nms_threshold)

        # Build canonical detections
        class_names = options.class_names or [str(i) for i in range(num_classes)]
        detections = []

        for idx in keep:
            cls_id = int(class_ids[idx])
            cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

            # Determine checked state from class name if available
            is_checked = None
            checkbox_state = "unknown"
            lower_name = cls_name.lower()
            if "unchecked" in lower_name:
                is_checked = False
                checkbox_state = "unchecked"
            elif "checked" in lower_name:
                is_checked = True
                checkbox_state = "checked"

            detections.append(
                {
                    "bbox": (float(x0[idx]), float(y0[idx]), float(x1[idx]), float(y1[idx])),
                    "coord_space": "image",
                    "confidence": float(scores[idx]),
                    "label": "checkbox",
                    "engine": "onnx",
                    "is_checked": is_checked,
                    "checkbox_state": checkbox_state,
                    "_original_class": cls_name,
                }
            )

        return detections
