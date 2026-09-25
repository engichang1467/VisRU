"""Lazy model loading.

ponytail: models live in this process, loaded on first use and cached. The README
plans a separate GPU worker; that buys batching and one load across clients, which
a single MCP server driving one agent does not need yet. Split it out when a second
client or concurrent tool calls make it hurt.
"""

import functools
import os

DETECT_MODEL = os.environ.get("VISRU_DETECT_MODEL", "google/owlv2-base-patch16-ensemble")
SEGMENT_MODEL = os.environ.get("VISRU_SEGMENT_MODEL", "facebook/sam2.1-hiera-small")
# Mobile detector, not server: paddle here is a CPU build, and the server detector on
# a 2000 px render is what the OOM killer comes for.
DETECT_TEXT_MODEL = os.environ.get("VISRU_TEXT_DETECT_MODEL", "PP-OCRv5_mobile_det")
RECOGNISE_TEXT_MODEL = os.environ.get("VISRU_TEXT_RECOGNISE_MODEL", "en_PP-OCRv5_mobile_rec")
TEXT_DET_SIDE = int(os.environ.get("VISRU_TEXT_DET_SIDE", "1280"))
EMBED_MODEL = os.environ.get("VISRU_EMBED_MODEL", "google/siglip-base-patch16-224")


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def dtype():
    import torch

    return torch.float16 if device() == "cuda" else torch.float32


def _pair(processor_cls: str, model_cls: str, name: str):
    """(processor, model) from transformers, on the best device, in eval mode."""
    import transformers

    processor = getattr(transformers, processor_cls).from_pretrained(name)
    model = getattr(transformers, model_cls).from_pretrained(name, dtype=dtype())
    return processor, model.to(device()).eval()


@functools.cache
def detector():
    """OWLv2: open-vocabulary boxes from text queries."""
    return _pair("Owlv2Processor", "Owlv2ForObjectDetection", DETECT_MODEL)


@functools.cache
def segmenter():
    """SAM 2: masks from points, boxes, or marks."""
    return _pair("Sam2Processor", "Sam2Model", SEGMENT_MODEL)


@functools.cache
def video_segmenter():
    """SAM 2 with memory: one prompt on one frame, masks propagated through the rest."""
    return _pair("Sam2VideoProcessor", "Sam2VideoModel", SEGMENT_MODEL)


class Embedder:
    """SigLIP image and text embeddings, L2-normalised so a dot product is cosine."""

    def __init__(self, name: str):
        from transformers import AutoModel, AutoProcessor

        self.name = name
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, dtype=dtype()).to(device()).eval()

    @staticmethod
    def _unit(features):
        # transformers 5 returns an output object from get_*_features; 4.x a tensor.
        features = getattr(features, "pooler_output", features)
        features = features.float()
        return (features / features.norm(dim=-1, keepdim=True)).cpu().numpy()

    def images(self, images):
        import torch

        inputs = self.processor(images=images, return_tensors="pt").to(device())
        with torch.no_grad():
            return self._unit(self.model.get_image_features(
                pixel_values=inputs["pixel_values"].to(dtype())))

    def text(self, text: str):
        import torch

        # SigLIP was trained on max_length padding; shorter padding skews the scores.
        inputs = self.processor(text=[text], padding="max_length",
                                return_tensors="pt").to(device())
        with torch.no_grad():
            return self._unit(self.model.get_text_features(**inputs))[0]


@functools.cache
def embedder() -> Embedder:
    """SigLIP: one vector per image or video keyframe, and per text query."""
    return Embedder(EMBED_MODEL)


@functools.cache
def reader():
    """PaddleOCR: text with boxes. Ships in the separate `ocr` extra."""
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "ocr needs the optional extra: uv pip install -e '.[ocr]' "
            "(kept separate because paddlepaddle's pins fight torch)") from exc

    # 3.x renamed use_angle_cls and dropped show_log; 2.x has neither new name.
    import inspect

    accepted = inspect.signature(PaddleOCR.__init__).parameters
    kwargs = {"lang": "en"}
    if "use_textline_orientation" in accepted:
        # 3.x ignores `lang` once any model name is set, so pin both explicitly
        # rather than let the recogniser fall back to the multilingual default.
        kwargs.pop("lang")
        kwargs.update(
            text_recognition_model_name=RECOGNISE_TEXT_MODEL,
            # paddlepaddle 3.3's oneDNN CPU kernels raise
            # "ConvertPirAttribute2RuntimeAttribute not support" on PP-OCRv5.
            # Plain CPU kernels work; drop this if a GPU build is installed.
            enable_mkldnn=False,
            # 3.x builds a five-model pipeline by default (doc orientation, doc
            # unwarping, textline orientation, server detector, recogniser) and gets
            # OOM-killed on a 2000 px render. visru already renders upright crops at
            # a known scale, so the three preprocessors have nothing to correct.
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_name=DETECT_TEXT_MODEL,
            text_det_limit_side_len=TEXT_DET_SIDE,
        )
    elif "use_angle_cls" in accepted:
        kwargs["use_angle_cls"] = True
    return PaddleOCR(**kwargs)


def unload() -> None:
    """Drop cached models. For tests and for freeing VRAM between heavy tools."""
    for fn in (detector, segmenter, video_segmenter, reader, embedder):
        fn.cache_clear()
