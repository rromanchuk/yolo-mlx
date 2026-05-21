# Copyright (c) 2026 webAI, Inc.
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO26 Model Class - Pure MLX Implementation

Main model class for YOLO26 detection, segmentation, pose, and OBB tasks.
Uses MLX v0.30.3 with full compilation support.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from yolo26mlx.converters.convert import convert_yolo26_weights
from yolo26mlx.engine.predictor import Predictor
from yolo26mlx.engine.trainer import Trainer
from yolo26mlx.engine.validator import Validator
from yolo26mlx.nn.tasks import build_model

logger = logging.getLogger(__name__)


class YOLO:
    """YOLO26 Model class for detection, segmentation, pose, and OBB tasks.

    This is the main interface for YOLO26 models in MLX. It supports:
    - Loading from YAML configs or pretrained weights
    - Inference with mx.compile optimization
    - Training with MLX optimizers
    - Weight conversion from PyTorch

    Example:
        >>> from yolo26mlx import YOLO
        >>> model = YOLO("models/yolo26n.npz")  # Load MLX weights (npz)
        >>> model = YOLO("models/yolo26n.safetensors")  # Load MLX weights (safetensors)
        >>> results = model.predict("image.jpg")
    """

    def __init__(
        self, model_path: str | Path | None = None, task: str = "detect", verbose: bool = True
    ):
        """Initialize YOLO26 model.

        Args:
            model_path: Path to model config (.yaml), MLX weights (.safetensors, .npz),
                        or PyTorch weights (.pt)
            task: Task type - 'detect', 'segment', 'pose', or 'obb'
            verbose: Print model information
        """
        self.model_path = Path(model_path) if model_path else None
        self.task = task
        self.verbose = verbose
        self.model: nn.Module | None = None
        self.predictor = None
        self.trainer = None
        self.validator = None
        self._tracker = None
        self._tracker_type = None

        # Model metadata
        self.names: dict[int, str] = {}
        self.nc: int = 80
        self.stride: mx.array | None = None

        if self.model_path:
            self._load_model()

    def _load_model(self):
        """Load model based on file extension."""
        if self.model_path is None:
            raise ValueError("No model path specified")

        self._auto_detect_task()

        suffix = self.model_path.suffix.lower()

        if suffix in (".yaml", ".yml"):
            self._build_from_yaml()
        elif suffix == ".safetensors":
            self._load_safetensors()
        elif suffix == ".npz":
            self._load_npz()
        elif suffix == ".pt":
            self._load_pytorch()
        else:
            raise ValueError(f"Unsupported model format: {suffix}")

    def _auto_detect_task(self):
        """Auto-detect task type from model filename if not explicitly set."""
        stem = self.model_path.stem.lower()
        if "-seg" in stem and self.task == "detect":
            self.task = "segment"
            if self.verbose:
                logger.info(f"Auto-detected task='segment' from filename '{self.model_path.name}'")

    def _build_from_yaml(self):
        """Build model from YAML configuration."""
        if self.verbose:
            logger.info(f"Building model from {self.model_path}")

        self.model = build_model(cfg=str(self.model_path), verbose=self.verbose)
        self._setup_metadata({})

    def _map_pytorch_to_mlx_name(self, pt_name: str) -> str:
        """Map PyTorch parameter name to MLX naming convention.

        Args:
            pt_name: Dot-separated PyTorch weight name (e.g. 'model.0.conv.weight').

        Returns:
            Equivalent MLX parameter name with layers/indices remapped.
        """
        name = pt_name

        # 1. Replace 'model.X.' with 'model.layers.X.'
        name = re.sub(r"^model\.(\d+)\.", r"model.layers.\1.", name)

        # 2. Handle layer 10 (C2PSA) - m.0. -> m.psa0.
        name = re.sub(r"layers\.10\.m\.(\d+)\.", r"layers.10.m.psa\1.", name)

        # 3. Handle layer 22 (C3k2 with attn) - m.0.X. -> m.0.layers.X.
        name = re.sub(r"layers\.22\.m\.0\.(\d+)\.", r"layers.22.m.0.layers.\1.", name)

        # 4. Handle layer 23 (Detect head) - cv2.X.Y. -> cv2.layerX.layers.Y.
        name = re.sub(r"layers\.23\.cv2\.(\d+)\.(\d+)\.", r"layers.23.cv2.layer\1.layers.\2.", name)

        # 5. Handle cv3 nested structure
        def map_cv3_nested(match):
            """Map cv3 nested block.layer indices to MLX flat layer index."""
            scale = match.group(1)
            block = int(match.group(2))
            layer = int(match.group(3))
            flat_idx = block * 2 + layer
            return f"layers.23.cv3.layer{scale}.layers.{flat_idx}."

        def map_cv3_final(match):
            """Map cv3 final-level block index to MLX flat layer index."""
            scale = match.group(1)
            block = int(match.group(2))
            return f"layers.23.cv3.layer{scale}.layers.{block * 2}."

        name = re.sub(r"layers\.23\.cv3\.(\d+)\.(\d+)\.(\d+)\.", map_cv3_nested, name)
        name = re.sub(r"layers\.23\.cv3\.(\d+)\.(\d+)\.", map_cv3_final, name)

        # 6. Handle one2one detection heads
        name = re.sub(
            r"layers\.23\.one2one_cv2\.(\d+)\.(\d+)\.",
            r"layers.23.one2one_cv2.layer\1.layers.\2.",
            name,
        )

        def map_one2one_cv3_nested(match):
            """Map one2one cv3 nested block.layer indices to MLX flat layer index."""
            scale = match.group(1)
            block = int(match.group(2))
            layer = int(match.group(3))
            flat_idx = block * 2 + layer
            return f"layers.23.one2one_cv3.layer{scale}.layers.{flat_idx}."

        def map_one2one_cv3_final(match):
            """Map one2one cv3 final-level block index to MLX flat layer index."""
            scale = match.group(1)
            block = int(match.group(2))
            return f"layers.23.one2one_cv3.layer{scale}.layers.{block * 2}."

        name = re.sub(
            r"layers\.23\.one2one_cv3\.(\d+)\.(\d+)\.(\d+)\.", map_one2one_cv3_nested, name
        )
        name = re.sub(r"layers\.23\.one2one_cv3\.(\d+)\.(\d+)\.", map_one2one_cv3_final, name)

        # 7. Segmentation head (cv4 + one2one_cv4 mask coefficient layers)
        name = re.sub(r"layers\.23\.cv4\.(\d+)\.(\d+)\.", r"layers.23.cv4.layer\1.layers.\2.", name)
        name = re.sub(
            r"layers\.23\.one2one_cv4\.(\d+)\.(\d+)\.",
            r"layers.23.one2one_cv4.layer\1.layers.\2.",
            name,
        )

        # 8. Proto26 multi-scale fusion (feat_refine ModuleList -> dict)
        name = re.sub(
            r"layers\.23\.proto\.feat_refine\.(\d+)\.",
            r"layers.23.proto.feat_refine.layer\1.",
            name,
        )

        # 9. Proto26 semseg Sequential (index -> layers.index)
        name = re.sub(
            r"layers\.23\.proto\.semseg\.(\d+)\.",
            r"layers.23.proto.semseg.layers.\1.",
            name,
        )

        return name

    def _get_param_names(self, params, prefix=""):
        """Recursively collect all dot-separated parameter names from a nested dict.

        Args:
            params: Nested dict/list of model parameters (from model.parameters()).
            prefix: Dot-separated prefix for current recursion level.

        Returns:
            Set of fully-qualified parameter name strings.
        """
        names = set()
        if isinstance(params, dict):
            for k, v in params.items():
                new_prefix = f"{prefix}.{k}" if prefix else k
                names.update(self._get_param_names(v, new_prefix))
        elif isinstance(params, list):
            for i, v in enumerate(params):
                new_prefix = f"{prefix}.{i}" if prefix else str(i)
                names.update(self._get_param_names(v, new_prefix))
        elif hasattr(params, "shape"):
            names.add(prefix)
        return names

    def _get_model_cfg(self) -> str:
        """Return the YAML config filename based on task type."""
        if self.task == "segment":
            return "yolo26-seg.yaml"
        return "yolo26.yaml"

    @staticmethod
    def _infer_scale_from_weights(weights: dict) -> str:
        """Infer model scale letter from weight tensor shapes.

        Reads the output-channel count of the first conv layer:
          96 → x, 64 → l or m (distinguished by a layer-2 block key), 32 → s, 16 → n.
        Falls back to 'n' if the key is absent (should not happen in practice).
        """
        l0 = weights.get("model.0.conv.weight")
        if l0 is None:
            return "n"
        ch = int(l0.shape[0])
        if ch == 96:
            return "x"
        if ch == 64:
            return "l" if "model.2.m.1.cv1.conv.weight" in weights else "m"
        if ch == 32:
            return "s"
        return "n"

    @staticmethod
    def _infer_nc_from_weights(weights: dict) -> int | None:
        """Infer number of classes from the Detect head cv3 output weight."""
        for k, v in weights.items():
            if re.search(r"model\.\d+\.cv3\.\d+\.2\.weight$", k):
                return int(v.shape[0])
        return None

    def _load_safetensors(self):
        """Load model weights from safetensors format."""
        if self.verbose:
            logger.info(f"Loading safetensors weights from {self.model_path}")

        # Read weights and embedded metadata in one pass
        raw_weights = dict(mx.load(str(self.model_path)))

        checkpoint_meta: dict = {}
        try:
            from safetensors import safe_open

            with safe_open(str(self.model_path), framework="numpy") as f:
                st_meta = f.metadata() or {}
            for k, v in st_meta.items():
                try:
                    checkpoint_meta[k] = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    checkpoint_meta[k] = v
        except ImportError:
            pass

        # scale: metadata → filename regex → shape inference
        scale = checkpoint_meta.get("scale")
        if not scale:
            match = re.search(r"yolo26([nsmlx])", self.model_path.stem)
            scale = match.group(1) if match else self._infer_scale_from_weights(raw_weights)

        # nc: metadata → shape inference
        nc = checkpoint_meta.get("nc") or self._infer_nc_from_weights(raw_weights)

        cfg = self._get_model_cfg()
        build_kwargs: dict = {"cfg": cfg, "verbose": self.verbose, "scale": scale}
        if nc is not None:
            build_kwargs["nc"] = nc
        try:
            self.model = build_model(**build_kwargs)
        except FileNotFoundError:
            self.model = build_model(cfg=cfg, verbose=self.verbose)

        mapped_weights = [(self._map_pytorch_to_mlx_name(k), v) for k, v in raw_weights.items()]
        mlx_param_names = self._get_param_names(self.model.parameters())
        matching_weights = [(k, v) for k, v in mapped_weights if k in mlx_param_names]

        if self.verbose:
            logger.info(f"  Matching weights: {len(matching_weights)}/{len(mapped_weights)}")

        self.model.load_weights(matching_weights, strict=False)
        self._setup_metadata(checkpoint_meta)

        if self.verbose:
            logger.info("Loaded weights successfully")

    def _load_npz(self):
        """Load model weights from npz format."""
        if self.verbose:
            logger.info(f"Loading npz weights from {self.model_path}")

        raw_weights = dict(mx.load(str(self.model_path)))

        # Read __metadata__ written by the converter (JSON-encoded string array)
        checkpoint_meta: dict = {}
        meta_arr = raw_weights.pop("__metadata__", None)
        if meta_arr is not None:
            try:
                checkpoint_meta = json.loads(str(meta_arr))
            except (json.JSONDecodeError, ValueError):
                pass

        # scale: metadata → shape inference (filename regex is unreliable)
        scale = checkpoint_meta.get("scale") or self._infer_scale_from_weights(raw_weights)

        # nc: metadata → shape inference
        nc = checkpoint_meta.get("nc") or self._infer_nc_from_weights(raw_weights)

        cfg = self._get_model_cfg()
        build_kwargs: dict = {"cfg": cfg, "verbose": self.verbose, "scale": scale}
        if nc is not None:
            build_kwargs["nc"] = nc
        try:
            self.model = build_model(**build_kwargs)
        except FileNotFoundError:
            self.model = build_model(cfg=cfg, verbose=self.verbose)

        mapped_weights = [(self._map_pytorch_to_mlx_name(k), v) for k, v in raw_weights.items()]
        mlx_param_names = self._get_param_names(self.model.parameters())
        matching_weights = [(k, v) for k, v in mapped_weights if k in mlx_param_names]

        if self.verbose:
            logger.info(f"  Matching weights: {len(matching_weights)}/{len(mapped_weights)}")

        self.model.load_weights(matching_weights, strict=False)
        self._setup_metadata(checkpoint_meta)

        if self.verbose:
            logger.info("Loaded weights successfully")

    def _load_pytorch(self):
        """Convert and load PyTorch weights."""
        if self.verbose:
            logger.info(f"Converting PyTorch weights from {self.model_path}")

        # Convert first so the safetensors header carries metadata
        output_path = self.model_path.with_suffix(".safetensors")
        convert_yolo26_weights(str(self.model_path), str(output_path), verbose=self.verbose)

        # Delegate to _load_safetensors which reads the embedded metadata
        orig_path = self.model_path
        self.model_path = output_path
        self._load_safetensors()
        self.model_path = orig_path

        if self.verbose:
            logger.info("Converted and loaded weights successfully")

    def _setup_metadata(self, checkpoint_meta: dict | None = None):
        """Setup model metadata (nc, stride, names) after loading."""
        if self.model is None:
            return
        meta = checkpoint_meta or {}
        self.nc = getattr(self.model, "nc", meta.get("nc", 80))
        self.stride = getattr(self.model, "stride", mx.array([8.0, 16.0, 32.0]))
        # Prefer names from checkpoint metadata (preserves actual class strings)
        if "names" in meta:
            self.names = {int(k): str(v) for k, v in meta["names"].items()}
        else:
            self.names = getattr(self.model, "names", {i: f"class{i}" for i in range(self.nc)})

    def predict(
        self,
        source: str | Path | list | Any,
        conf: float = 0.25,
        imgsz: int = 640,
        save: bool = False,
        stream: bool = False,
        rect: bool = True,
    ):
        """Run inference on image(s).

        Args:
            source: Image path, directory, URL, PIL Image, or numpy array
            conf: Confidence threshold for detections
            imgsz: Input image size
            save: Save results to disk
            stream: Return a generator for large batches
            rect: Rectangular inference (pad to stride-aligned dims, not square).
                  Default True to match ultralytics behavior. Faster for non-square images.

        Returns:
            List of Results objects
        """
        if self.predictor is None:
            self.predictor = Predictor(
                model=self.model, task=self.task, names=self.names, stride=self.stride
            )

        return self.predictor(
            source=source, conf=conf, imgsz=imgsz, save=save, stream=stream, rect=rect
        )

    def train(
        self,
        data: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        patience: int = 50,
        save_period: int = -1,
        project: str = "runs/train",
        name: str = "exp",
        exist_ok: bool = False,
        resume: bool = False,
        val: bool = True,
        verbose: bool = True,
    ):
        """Train the model.

        Args:
            data: Path to data configuration file (YAML)
            epochs: Number of training epochs
            imgsz: Input image size
            batch: Batch size
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs (-1 to disable)
            project: Project directory
            name: Experiment name
            exist_ok: Overwrite existing experiment
            resume: Resume from last checkpoint
            val: Run validation after each epoch (default: True). Set False
                for throughput benchmarks where validation cost is irrelevant.
            verbose: Print per-batch and per-epoch progress (default: True).

        Returns:
            Training results dict
        """
        if self.trainer is None:
            self.trainer = Trainer(model=self.model, task=self.task)

        return self.trainer(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            save_period=save_period,
            project=project,
            name=name,
            exist_ok=exist_ok,
            resume=resume,
            val=val,
            verbose=verbose,
        )

    def val(
        self,
        data: str | None = None,
        batch: int = 16,
        imgsz: int = 640,
        conf: float = 0.001,
        iou: float = 0.6,
    ):
        """Validate the model.

        Args:
            data: Path to validation data configuration
            batch: Batch size
            imgsz: Input image size
            conf: Confidence threshold
            iou: IoU threshold

        Returns:
            Validation metrics dict
        """
        if self.validator is None:
            self.validator = Validator(model=self.model, task=self.task)

        return self.validator(data=data, batch=batch, imgsz=imgsz, conf=conf, iou=iou)

    def info(self, verbose: bool = True) -> dict:
        """Get model information.

        Args:
            verbose: Print to console

        Returns:
            Model info dict
        """
        if self.model is None:
            return {"status": "Model not loaded"}

        # Count parameters
        def count_params(params):
            """Recursively count total number of scalar parameters in a nested dict."""
            total = 0
            for v in params.values():
                if isinstance(v, dict):
                    total += count_params(v)
                elif hasattr(v, "size"):
                    total += v.size
            return total

        n_params = count_params(self.model.parameters())
        n_layers = len(self.model.model) if hasattr(self.model, "model") else 0

        info = {
            "task": self.task,
            "nc": self.nc,
            "layers": n_layers,
            "parameters": n_params,
            "stride": list(map(float, self.stride)) if self.stride is not None else None,
        }

        if verbose:
            logger.info("YOLO26 Model Info:")
            logger.info(f"  Task: {info['task']}")
            logger.info(f"  Classes: {info['nc']}")
            logger.info(f"  Layers: {info['layers']}")
            logger.info(f"  Parameters: {info['parameters']:,}")
            logger.info(f"  Stride: {info['stride']}")

        return info

    def save(self, path: str | Path, format: str = "safetensors"):
        """Save model weights.

        Args:
            path: Output path
            format: Output format ('safetensors' or 'npz')
        """
        if self.model is None:
            raise ValueError("No model to save")

        path = Path(path)
        if format == "safetensors":
            path = path.with_suffix(".safetensors")
        else:
            path = path.with_suffix(".npz")

        self.model.save_weights(str(path))

        if self.verbose:
            logger.info(f"Saved model to {path}")

    def track(
        self,
        source,
        tracker: str = "bytetrack.yaml",
        conf: float = 0.25,
        imgsz: int = 640,
        persist: bool = False,
        stream: bool = False,
        show: bool = False,
        save: bool = False,
        vid_stride: int = 1,
    ):
        """Run tracking on video/image source.

        Args:
            source: Video path, webcam index (0), image path, or numpy array.
            tracker: Tracker config YAML filename (bytetrack.yaml or botsort.yaml).
            conf: Confidence threshold.
            imgsz: Input image size.
            persist: Keep tracker state between calls (for frame-by-frame API).
            stream: Return generator instead of list.
            show: Display results with cv2.imshow.
            save: Save annotated video to disk.
            vid_stride: Process every Nth frame.

        Returns:
            list[Results] with track IDs populated in boxes.id.
        """
        from yolo26mlx.engine.tracker import TrackerManager

        # (Re)create tracker if needed
        if not persist or self._tracker is None or self._tracker_type != tracker:
            self._tracker = TrackerManager(tracker)
            self._tracker_type = tracker

        is_video = isinstance(source, str) and not self._is_image_path(source)
        is_webcam = isinstance(source, int)
        is_frame = hasattr(source, "shape") and len(source.shape) == 3  # numpy array

        # --- Single frame (numpy array) ---
        if is_frame:
            det_results = self.predict(source, conf=conf, imgsz=imgsz)
            tracked = self._tracker.update(det_results[0])
            return [tracked]

        # --- Video or webcam ---
        if is_video or is_webcam:
            return self._track_video(
                source,
                conf=conf,
                imgsz=imgsz,
                stream=stream,
                show=show,
                save=save,
                vid_stride=vid_stride,
            )

        # --- Image file / directory (single-frame tracking) ---
        det_results = self.predict(source, conf=conf, imgsz=imgsz)
        all_tracked = []
        for r in det_results:
            tracked = self._tracker.update(r)
            all_tracked.append(tracked)
        return all_tracked

    def _track_video(
        self,
        source,
        conf: float = 0.25,
        imgsz: int = 640,
        stream: bool = False,
        show: bool = False,
        save: bool = False,
        vid_stride: int = 1,
    ):
        """Run tracking on a video or webcam source.

        Args:
            source: Video file path or webcam index (int).
            conf: Confidence threshold for detections.
            imgsz: Input image size for the model.
            stream: If True, return a generator; otherwise collect all results.
            show: Display annotated frames with cv2.imshow.
            save: Save annotated video to results/<name>_tracked.mp4.
            vid_stride: Process every Nth frame.

        Returns:
            list[Results] or Generator[Results]: Tracked results per frame.
        """
        from yolo26mlx.engine.tracker import TrackerManager
        from yolo26mlx.utils.video import VideoSource, VideoWriter

        src = VideoSource(source, vid_stride=vid_stride)
        frame_rate = max(1, int(src.fps))
        # Reinitialize tracker with video frame rate
        self._tracker = TrackerManager(self._tracker_type, frame_rate=frame_rate)

        writer = None
        if save:
            os.makedirs("results", exist_ok=True)
            src_name = str(source) if not isinstance(source, int) else "webcam"
            out_name = os.path.join(
                "results", os.path.splitext(os.path.basename(src_name))[0] + "_tracked.mp4"
            )
            writer = VideoWriter(out_name, fps=src.fps, width=src.width, height=src.height)

        def _gen():
            import cv2

            try:
                for frame in src:
                    det_results = self.predict(frame, conf=conf, imgsz=imgsz)
                    tracked = self._tracker.update(det_results[0])

                    if show:
                        annotated = tracked.plot()
                        # plot() returns RGB; imshow expects BGR
                        cv2.imshow("YOLO26 Tracking", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                    if writer is not None:
                        annotated = tracked.plot()
                        # plot() returns RGB; VideoWriter expects BGR
                        writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

                    yield tracked
            finally:
                src.release()
                if writer is not None:
                    writer.release()
                if show:
                    cv2.destroyAllWindows()

        if stream:
            return _gen()
        return list(_gen())

    @staticmethod
    def _is_image_path(path: str) -> bool:
        """Check if a path looks like an image file or directory.

        Args:
            path: File or directory path to check.

        Returns:
            True if the path has a common image extension or is a directory.
        """
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
        p = Path(path)
        return p.suffix.lower() in img_exts or p.is_dir()

    def __call__(self, source, **kwargs):
        """Run inference on source image(s), forwarding to predict().

        Args:
            source: Image path, numpy array, PIL Image, or list of sources.
            **kwargs: Additional arguments passed to predict().

        Returns:
            list[Results]: Detection results for each input image.
        """
        return self.predict(source, **kwargs)
