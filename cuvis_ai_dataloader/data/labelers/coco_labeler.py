"""COCO label parsing + rasterization. Not a plugin contract.

A typed view over a ``pycocotools`` COCO file (``COCOData``) plus the ``CocoLabeler`` used
by the cu3s DataModules to turn per-image annotations into category-id masks.
"""

import contextlib
import io
import json
from collections.abc import Iterable, Iterator
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dataclass_wizard import JSONWizard
from loguru import logger
from pycocotools.coco import COCO
from skimage.draw import polygon2mask
from torchvision.tv_tensors import BoundingBoxes, Mask

from cuvis_ai_core.data.rle import (
    coco_rle_area,
    coco_rle_decode,
    coco_rle_to_bbox,
    decode_rle_mask_for_canvas,
)


class SafeWizard(JSONWizard):
    """
    JSONWizard subclass that safely converts dataclasses to dicts,
    keeping non-serializable objects (e.g., torch Tensors, Masks)
    as-is instead of falling back to string representations.
    """

    def to_dict_safe(self) -> dict[str, Any]:
        """
        Like `to_dict()`, but leaves unsupported types untouched.
        """
        base_dict = super().to_dict()
        final_dict = {}

        for key, value in vars(self).items():
            if not self._is_json_serializable(value):
                # keep original object (Mask, Tensor, etc.)
                final_dict[key] = value
                continue
            val = base_dict.get(key, value)
            final_dict[key] = val
        return final_dict

    @staticmethod
    def _is_json_serializable(obj):
        try:
            json.dumps(obj)
            return True
        except Exception:
            return False


def _decode_rle_dict_for_canvas(seg: dict, image_height: int, image_width: int) -> np.ndarray:
    """Decode a standard COCO RLE ``segmentation`` dict to a boolean mask on the canvas.

    List-based counts decode straight into the canvas dimensions (matching the legacy
    ``mask``-key behavior); compressed string counts decode at their declared ``size`` and
    are padded/cropped to the canvas when the two disagree.
    """
    counts = seg.get("counts")
    if isinstance(counts, list):
        return decode_rle_mask_for_canvas(
            seg, target_height=image_height, target_width=image_width
        ).astype(bool, copy=False)
    decoded = coco_rle_decode(seg).astype(bool, copy=False)
    if decoded.shape != (image_height, image_width):
        logger.warning(
            "COCO RLE size {} mismatches canvas {}; padding/cropping to the canvas.",
            tuple(decoded.shape),
            (image_height, image_width),
        )
        fitted = np.zeros((image_height, image_width), dtype=bool)
        h = min(image_height, decoded.shape[0])
        w = min(image_width, decoded.shape[1])
        fitted[:h, :w] = decoded[:h, :w]
        return fitted
    return decoded


def convert_track_dialect_to_image(dataset: dict, source: str = "<memory>") -> dict:
    """Convert a track-dialect COCO into standard image-keyed COCO, in memory.

    The track dialect (top-level ``videos``, one annotation per track carrying per-frame
    parallel arrays ``segmentations``/``bboxes``/``areas``/``detection_scores``, no
    ``image_id``) is what the mask-tracking tools historically wrote. The output is standard
    COCO — per-frame ``images`` records plus one annotation per (track, frame) — with object
    identity preserved as an additive ``track_id`` per annotation.

    Malformed or ambiguous inputs are rejected loudly instead of guessed at: hybrid files
    carrying both ``videos`` and ``images``, empty or multi-entry ``videos``, duplicate
    ``frame_indices``, parallel arrays whose lengths disagree with ``frame_indices``, and
    non-RLE segmentation entries all raise ``ValueError`` naming ``source``.
    """
    if "images" in dataset:
        raise ValueError(
            f"{source}: hybrid COCO carrying both 'videos' and 'images' is ambiguous; "
            "re-export the file in a single dialect."
        )
    videos = dataset.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError(f"{source}: track-dialect COCO has an empty or invalid 'videos' list.")
    if len(videos) != 1:
        raise ValueError(
            f"{source}: track-dialect COCO with {len(videos)} videos is unsupported "
            "(expected exactly one)."
        )
    video = videos[0]
    frame_indices = video.get("frame_indices")
    if not isinstance(frame_indices, list) or not frame_indices:
        raise ValueError(f"{source}: track-dialect video record has no 'frame_indices'.")
    frame_ids = [int(fid) for fid in frame_indices]
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError(f"{source}: track-dialect 'frame_indices' contains duplicates.")
    try:
        height, width = int(video["height"]), int(video["width"])
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError(
            f"{source}: track-dialect video record lacks integer height/width."
        ) from err
    if height <= 0 or width <= 0:
        raise ValueError(f"{source}: track-dialect video record has non-positive dimensions.")

    n_frames = len(frame_ids)
    images = [
        {"id": fid, "file_name": f"frame_{fid:06d}", "height": height, "width": width}
        for fid in frame_ids
    ]
    annotations: list[dict[str, Any]] = []
    next_id = 1
    for ann in dataset.get("annotations", []):
        segmentations = ann.get("segmentations")
        if not isinstance(segmentations, list) or len(segmentations) != n_frames:
            found = len(segmentations) if isinstance(segmentations, list) else "no"
            raise ValueError(
                f"{source}: track annotation {ann.get('id')} carries {found} "
                f"'segmentations' entries for {n_frames} frames."
            )
        per_frame: dict[str, list] = {}
        for key in ("bboxes", "areas", "detection_scores"):
            values = ann.get(key)
            if values is None:
                continue
            if not isinstance(values, list) or len(values) != n_frames:
                found = len(values) if isinstance(values, list) else "an invalid"
                raise ValueError(
                    f"{source}: track annotation {ann.get('id')} carries {found} "
                    f"'{key}' entries for {n_frames} frames."
                )
            per_frame[key] = values
        track_id = int(ann.get("track_id", ann.get("id", -1)))
        category_id = int(ann.get("category_id", 1))
        for i, fid in enumerate(frame_ids):
            seg = segmentations[i]
            if seg is None:
                continue
            if not isinstance(seg, dict) or "counts" not in seg:
                raise ValueError(
                    f"{source}: track annotation {ann.get('id')} frame {fid} has a "
                    "non-RLE segmentation entry (expected an RLE object or null)."
                )
            bbox = per_frame.get("bboxes", [None] * n_frames)[i]
            area = per_frame.get("areas", [None] * n_frames)[i]
            score = per_frame.get("detection_scores", [None] * n_frames)[i]
            converted: dict[str, Any] = {
                "id": next_id,
                "image_id": fid,
                "category_id": category_id,
                "segmentation": seg,
                "bbox": [float(v) for v in bbox] if bbox is not None else coco_rle_to_bbox(seg),
                "area": float(area) if area is not None else float(coco_rle_area(seg)),
                "iscrowd": 1,
                "track_id": track_id,
            }
            if score is not None:
                converted["score"] = float(score)
            annotations.append(converted)
            next_id += 1

    return {
        "info": dataset.get("info", {}),
        "licenses": dataset.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": dataset.get("categories", []),
    }


@dataclass
class Info(JSONWizard):
    """COCO ``info`` block: free-form dataset description / version metadata."""

    description: str | None = None
    url: str | None = None
    version: int | None = None
    contributor: str | None = None
    date_created: str | None = None


@dataclass
class License(JSONWizard):
    """COCO ``license`` entry."""

    id: int
    name: str
    url: str | None = None


@dataclass
class Category(JSONWizard):
    """COCO ``category``: id, name, and optional supercategory."""

    id: int
    name: str
    supercategory: str | None = None


@dataclass
class Image(JSONWizard):
    """COCO ``image`` record: id, file name, size, and optional per-band wavelengths."""

    id: int
    file_name: str
    height: int
    width: int
    license: int | None = None
    flickr_url: str | None = None
    coco_url: str | None = None
    date_captured: str | None = None
    wavelength: list[float] | None = field(default_factory=list)


@dataclass
class Annotation(SafeWizard):
    """COCO ``annotation``: bbox / polygon / RLE mask for one image and category.

    ``segmentation`` accepts the standard COCO forms: polygon list-of-lists or an RLE
    object (``{"size": [H, W], "counts": str | list}``). The legacy non-standard ``mask``
    key (list-counts RLE) is kept for older exports.
    """

    id: int
    image_id: int
    category_id: int
    segmentation: list | dict | None = None
    area: float | None = None
    bbox: list[float] | None = None
    mask: dict | None = None
    iscrowd: int | None = 0
    auxiliary: dict[str, Any] | None = field(default_factory=dict)

    def to_torchvision(self, size: tuple[int, int]) -> dict[str, Any]:
        """Convert COCO-style bbox/segmentation/mask into torchvision tensors."""
        out = copy(self)
        canvas_height, canvas_width = int(size[0]), int(size[1])

        if self.bbox is not None:
            out.bbox = BoundingBoxes(
                torch.tensor([self.bbox], dtype=torch.float32),
                format="XYWH",
                canvas_size=size,
            )

        if (
            self.segmentation is not None
            and isinstance(self.segmentation, list)
            and self.segmentation != []
        ):
            coords = np.array(self.segmentation[0]).reshape(-1, 2)
            mask_np = polygon2mask(size, coords).astype(np.uint8)
            out.segmentation = Mask(torch.from_numpy(mask_np))
        elif isinstance(self.segmentation, dict) and self.segmentation.get("counts") is not None:
            mask_np = _decode_rle_dict_for_canvas(
                self.segmentation, canvas_height, canvas_width
            ).astype(np.uint8)
            out.segmentation = Mask(torch.from_numpy(mask_np))

        if self.mask is not None:
            mask_np = decode_rle_mask_for_canvas(
                self.mask,
                target_height=canvas_height,
                target_width=canvas_width,
            )
            out.mask = Mask(torch.from_numpy(mask_np))

        return out.to_dict_safe()


class QueryableList:
    """A list wrapper with a ``where(**conditions)`` attribute-equality filter."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def where(self, **conditions) -> list[Any]:
        """
        Filter items based on conditions.
        :param conditions: Keyword arguments representing field=value filters.
        :return: A new QueryableList with filtered items.
        """
        filtered_items = self._items
        for key, value in conditions.items():
            filtered_items = [item for item in filtered_items if getattr(item, key) == value]
        return list(filtered_items)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Any:
        return self._items[index]


class COCOData:
    """A typed, lazily-parsed view over a ``pycocotools`` ``COCO`` object."""

    def __init__(self, coco: COCO) -> None:
        self._coco = coco
        self._image_ids: list[int] | None = None
        self._categories: list[Category] | None = None
        self._category_id_to_name: dict[int, str] | None = None
        self._annotations: QueryableList | None = None
        self._images: list[Image] | None = None

    @classmethod
    def from_path(cls, path: Path | str):
        """Load a COCO JSON from ``path`` (suppressing pycocotools' stdout noise).

        Both label dialects are accepted: standard image-keyed COCO loads as-is, and a
        track-dialect file (top-level ``videos``, per-track parallel arrays) is converted
        to the image dialect in memory via :func:`convert_track_dialect_to_image` before
        pycocotools indexes it.
        """
        with open(path, encoding="utf-8") as f:
            dataset = json.load(f)
        if not isinstance(dataset, dict):
            raise ValueError(f"{path}: COCO file does not contain a JSON object.")
        if "videos" in dataset:
            dataset = convert_track_dialect_to_image(dataset, source=str(path))
        coco = COCO()
        coco.dataset = dataset
        with contextlib.redirect_stdout(io.StringIO()):
            coco.createIndex()
        return cls(coco)

    @property
    def image_ids(self) -> list[int]:
        """Sorted list of COCO image ids."""
        if self._image_ids is None:
            self._image_ids = sorted(self._coco.imgs.keys())
        return self._image_ids

    @property
    def info(self) -> Info:
        """COCO ``info`` block."""
        return Info.from_dict(self._coco.dataset["info"])

    @property
    def license(self) -> License:
        """First COCO ``license`` entry."""
        return License.from_dict(self._coco.dataset["licenses"][0])

    @property
    def annotations(self) -> QueryableList:
        """All annotations as a queryable list."""
        if self._annotations is None:
            self._annotations = QueryableList(
                [Annotation.from_dict(v) for v in self._coco.anns.values()]
            )
        return self._annotations

    @property
    def categories(self) -> list[Category]:
        """COCO categories."""
        if self._categories is None:
            self._categories = [Category.from_dict(v) for v in self._coco.cats.values()]
        return self._categories

    @property
    def category_id_to_name(self) -> dict[int, str]:
        """Mapping of category id to category name."""
        if self._category_id_to_name is None:
            self._category_id_to_name = {cat.id: cat.name for cat in self.categories}
        return self._category_id_to_name

    @property
    def images(self) -> list[Image]:
        """COCO image records."""
        if self._images is None:
            self._images = [Image.from_dict(v) for v in self._coco.imgs.values()]
        return self._images

    def save(self, path: str | Path) -> None:
        """
        Save the current COCOData object (images, annotations, categories, etc.)
        back into a COCO-style JSON file.

        Automatically converts dataclasses to plain dicts and ensures
        compliance with standard COCO structure.
        """
        path = str(path)
        annotations_list: list[dict[str, Any]] = []

        ann: Annotation | dict[str, Any]
        for ann in self.annotations:
            if isinstance(ann, Annotation):
                annotations_list.append(ann.to_dict_safe())
            elif isinstance(ann, dict):
                annotations_list.append(ann)
            else:
                raise TypeError(f"Unsupported annotation type: {type(ann)}")

        dataset = {
            "info": self.info.to_dict() if hasattr(self, "info") else {},
            "licenses": self._coco.dataset.get("licenses", []),
            "images": [img.to_dict() for img in self.images],
            "annotations": annotations_list,
            "categories": [cat.to_dict() for cat in self.categories],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)

        logger.debug(f"COCOData saved to: {path}")


def create_mask(
    annotations: Iterable[Annotation],
    image_height: int,
    image_width: int,
    overlap_strategy: str = "overwrite",
) -> np.ndarray:
    """Rasterize COCO annotations into a per-pixel category-id mask.

    Polygons are filled via ``skimage.draw.polygon2mask``; standard RLE-object
    ``segmentation`` dicts (str or list counts) are decoded via core's RLE helpers, as is
    the legacy non-standard ``mask`` key. Returns an ``int32`` ``[H, W]`` array of
    category ids (0 = background).
    """
    category_mask = np.zeros((image_height, image_width), dtype=np.int32)
    for ann in annotations:
        segs = ann.segmentation
        mask = ann.mask
        cat_id = int(ann.category_id)
        if not segs and not mask:
            continue

        if isinstance(segs, dict) and segs.get("counts") is not None:
            decoded = _decode_rle_dict_for_canvas(segs, image_height, image_width)
            if overlap_strategy == "overwrite":
                write_mask = decoded
            else:
                write_mask = decoded & (category_mask == 0)
            category_mask[write_mask] = cat_id
        elif isinstance(segs, list) and len(segs) > 0 and isinstance(segs[0], (list, tuple)):
            for seg in segs:
                if len(seg) < 6:
                    continue
                xy = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
                # polygon2mask expects (row, col); swap x,y.
                poly_mask = polygon2mask((image_height, image_width), xy[:, [1, 0]])
                if overlap_strategy == "overwrite":
                    category_mask[poly_mask] = cat_id
                else:
                    write_idx = poly_mask & (category_mask == 0)
                    category_mask[write_idx] = cat_id
        counts = mask.get("counts") if isinstance(mask, dict) else None
        if counts is not None and len(counts) > 0:
            decoded = decode_rle_mask_for_canvas(
                mask, target_height=image_height, target_width=image_width
            )
            if overlap_strategy == "overwrite":
                write_mask = decoded
            else:
                write_mask = decoded & (category_mask == 0)
            category_mask[write_mask] = cat_id

    return category_mask


class CocoLabeler:
    """Caches one parsed COCO file and rasterizes per-image category masks.

    Used by the cu3s DataModules: one labeler per unique annotation JSON. Keys purely on the
    COCO ``image_id``. For per-frame cu3s that id is the measurement index; for *merged* cu3s
    with sparse annotations it is not, so the converter passes an explicit ``image_id`` per read
    frame (see ``npz_converter.convert_cu3s_file``'s ``image_ids``).
    """

    def __init__(self, annotation_json_path: str | Path) -> None:
        self.annotation_json_path = str(annotation_json_path)
        self._coco = COCOData.from_path(self.annotation_json_path)
        self.category_id_to_name = self._coco.category_id_to_name

    @property
    def image_ids(self) -> list[int]:
        return self._coco.image_ids

    def is_annotated(self, image_id: int) -> bool:
        """True if ``image_id`` exists in the COCO and carries at least one annotation."""
        if image_id not in self._coco.image_ids:
            return False
        return bool(self._coco.annotations.where(image_id=image_id))

    def categories_for(self, image_id: int) -> list[int]:
        """Distinct category ids annotated on ``image_id`` (empty -> unannotated / normal)."""
        if image_id not in self._coco.image_ids:
            return []
        seen: list[int] = []
        for ann in self._coco.annotations.where(image_id=image_id):
            cid = int(ann.category_id)
            if cid not in seen:
                seen.append(cid)
        return seen

    def _canvas_size(self, image_id: int, fallback_hw: tuple[int, int]) -> tuple[int, int]:
        """COCO image (height, width) for ``image_id``; falls back to the cube's.

        A COCO ``image`` record whose height or width is zero/negative (some exporters leave
        these unset — e.g. the lentils day-level COCOs ship ``height=0, width=0``) is treated
        as absent and falls back to the cube size, so masks rasterize at the real frame
        resolution instead of collapsing to 0x0.
        """
        fb_h, fb_w = int(fallback_hw[0]), int(fallback_hw[1])
        images = getattr(self._coco, "images", None)
        if isinstance(images, list):
            for image in images:
                if getattr(image, "id", None) != image_id:
                    continue
                try:
                    h, w = int(image.height), int(image.width)
                    if h > 0 and w > 0:
                        return h, w
                except (AttributeError, TypeError, ValueError):
                    pass
                break
        coco_backend = getattr(self._coco, "_coco", None)
        image_lookup = getattr(coco_backend, "imgs", None)
        if isinstance(image_lookup, dict):
            meta = image_lookup.get(image_id)
            if isinstance(meta, dict):
                try:
                    h, w = int(meta["height"]), int(meta["width"])
                    if h > 0 and w > 0:
                        return h, w
                except (KeyError, TypeError, ValueError):
                    pass
        return fb_h, fb_w

    def load_for(self, image_id: int, item: dict) -> dict:
        """Return ``{"mask": int32[H,W]}`` for ``image_id`` (zeros if unannotated)."""
        cube = item["cube"]
        fb_hw = (cube.shape[0], cube.shape[1])
        if image_id in self._coco.image_ids:
            anns = self._coco.annotations.where(image_id=image_id)
            json_h, json_w = self._canvas_size(image_id, fb_hw)
            mask = create_mask(annotations=anns, image_height=json_h, image_width=json_w)
        else:
            mask = np.zeros((int(fb_hw[0]), int(fb_hw[1])), dtype=np.int32)
        return {"mask": mask}
