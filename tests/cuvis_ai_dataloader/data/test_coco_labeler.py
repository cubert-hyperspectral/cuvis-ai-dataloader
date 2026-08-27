"""Tests for COCO labels data structures and utilities."""

import json
from pathlib import Path

import pytest
import torch
from pycocotools.coco import COCO

from cuvis_ai_dataloader.data.labelers.coco_labeler import (
    Annotation,
    Category,
    COCOData,
    Image,
    Info,
    License,
    QueryableList,
)
from cuvis_ai_core.data.rle import rle_list_to_mask


@pytest.fixture
def minimal_coco_json():
    """Create a minimal COCO format JSON for testing."""
    return {
        "info": {
            "description": "Test Dataset",
            "url": "https://test.com",
            "version": 1,
            "contributor": "Test User",
            "date_created": "2026-02-09",
        },
        "licenses": [{"id": 1, "name": "Test License", "url": "https://test.com/license"}],
        "images": [
            {
                "id": 1,
                "file_name": "test_image_1.jpg",
                "width": 640,
                "height": 480,
                "license": 1,
            },
            {
                "id": 2,
                "file_name": "test_image_2.jpg",
                "width": 800,
                "height": 600,
            },
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100.0, 100.0, 50.0, 50.0],
                "area": 2500.0,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "segmentation": [[10, 10, 10, 20, 20, 20, 20, 10]],
                "area": 100.0,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "person", "supercategory": "human"},
            {"id": 2, "name": "car", "supercategory": "vehicle"},
        ],
    }


@pytest.fixture
def coco_json_file(minimal_coco_json, tmp_path):
    """Create a temporary COCO JSON file."""
    json_path = tmp_path / "test_coco.json"
    with open(json_path, "w") as f:
        json.dump(minimal_coco_json, f)
    return json_path


@pytest.fixture
def coco_data(coco_json_file):
    """Create a COCOData instance from the test file."""
    return COCOData.from_path(coco_json_file)


class TestCOCODataInitialization:
    """Test COCOData initialization and type annotations."""

    def test_init_type_annotations(self, coco_json_file):
        """Test that __init__ properly initializes typed attributes."""
        coco_data = COCOData.from_path(coco_json_file)

        # Check that type-annotated attributes are initialized to None
        assert coco_data._image_ids is None
        assert coco_data._categories is None
        assert coco_data._category_id_to_name is None
        assert coco_data._annotations is None
        assert coco_data._images is None

    def test_from_path_classmethod(self, coco_json_file):
        """Test from_path classmethod creates COCOData instance."""
        coco_data = COCOData.from_path(coco_json_file)
        assert isinstance(coco_data, COCOData)
        assert coco_data._coco is not None
        assert isinstance(coco_data._coco, COCO)


class TestCOCODataProperties:
    """Test COCOData lazy-loaded properties."""

    def test_image_ids_property(self, coco_data):
        """Test image_ids property returns sorted list."""
        image_ids = coco_data.image_ids
        assert isinstance(image_ids, list)
        assert len(image_ids) == 2
        assert image_ids == [1, 2]
        assert image_ids == sorted(image_ids)

    def test_info_property(self, coco_data):
        """Test info property returns Info dataclass."""
        info = coco_data.info
        assert isinstance(info, Info)
        assert info.description == "Test Dataset"
        assert info.version == 1

    def test_license_property(self, coco_data):
        """Test license property returns License (not Info)."""
        # This tests the bug fix where it was returning Info.from_dict
        license_obj = coco_data.license
        assert isinstance(license_obj, License)
        assert license_obj.id == 1
        assert license_obj.name == "Test License"

    def test_categories_property(self, coco_data):
        """Test categories property returns list of Category objects."""
        categories = coco_data.categories
        assert isinstance(categories, list)
        assert len(categories) == 2
        assert all(isinstance(cat, Category) for cat in categories)
        assert categories[0].name in ["person", "car"]

    def test_category_id_to_name_property(self, coco_data):
        """Test category_id_to_name property returns correct mapping."""
        mapping = coco_data.category_id_to_name
        assert isinstance(mapping, dict)
        assert mapping[1] in ["person", "car"]
        assert mapping[2] in ["person", "car"]
        assert len(mapping) == 2

    def test_images_property(self, coco_data):
        """Test images property returns list of Image objects."""
        images = coco_data.images
        assert isinstance(images, list)
        assert len(images) == 2
        assert all(isinstance(img, Image) for img in images)

    def test_annotations_property(self, coco_data):
        """Test annotations property returns QueryableList."""
        annotations = coco_data.annotations
        assert isinstance(annotations, QueryableList)
        assert len(annotations) == 2


class TestCOCODataSave:
    """Test COCOData save method with new type-safe implementation."""

    def test_save_creates_file(self, coco_data, tmp_path):
        """Test that save creates a valid JSON file."""
        output_path = tmp_path / "output_coco.json"
        coco_data.save(output_path)

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)

        assert "info" in data
        assert "images" in data
        assert "annotations" in data
        assert "categories" in data

    def test_save_preserves_annotations(self, coco_data, tmp_path):
        """Test that save preserves all annotations correctly."""
        output_path = tmp_path / "output_coco.json"
        coco_data.save(output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert len(data["annotations"]) == 2
        assert data["annotations"][0]["id"] == 1
        assert data["annotations"][1]["id"] == 2

    def test_save_handles_annotation_objects(self, coco_data, tmp_path):
        """Test save handles Annotation dataclass instances."""
        # Access annotations to ensure they're loaded as Annotation objects
        annotations = coco_data.annotations
        assert len(annotations) > 0
        assert isinstance(annotations[0], Annotation)

        output_path = tmp_path / "output_coco.json"
        coco_data.save(output_path)

        with open(output_path) as f:
            data = json.load(f)

        # Verify annotations were properly serialized
        assert isinstance(data["annotations"], list)
        assert len(data["annotations"]) == 2

    def test_save_with_path_string(self, coco_data, tmp_path):
        """Test save accepts string path."""
        output_path = str(tmp_path / "output_string.json")
        coco_data.save(output_path)
        assert Path(output_path).exists()

    def test_save_with_path_object(self, coco_data, tmp_path):
        """Test save accepts Path object."""
        output_path = tmp_path / "output_path.json"
        coco_data.save(output_path)
        assert output_path.exists()

    def test_save_handles_dict_annotations(self, coco_data, tmp_path):
        """Test save handles raw dict annotations (not Annotation objects)."""
        dict_ann = {"id": 99, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]}
        coco_data._annotations = QueryableList([dict_ann])

        output_path = tmp_path / "output_dict_ann.json"
        coco_data.save(output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert len(data["annotations"]) == 1
        assert data["annotations"][0]["id"] == 99


class TestQueryableList:
    """Test QueryableList filtering functionality."""

    def test_where_filters_by_single_condition(self, coco_data):
        """Test where method filters items by single attribute."""
        annotations = coco_data.annotations
        filtered = annotations.where(image_id=1)

        assert len(filtered) == 2
        assert all(ann.image_id == 1 for ann in filtered)

    def test_where_filters_by_multiple_conditions(self, coco_data):
        """Test where method filters by multiple attributes."""
        annotations = coco_data.annotations
        filtered = annotations.where(image_id=1, category_id=1)

        assert len(filtered) == 1
        assert filtered[0].image_id == 1
        assert filtered[0].category_id == 1

    def test_queryable_list_iteration(self, coco_data):
        """Test QueryableList is iterable."""
        annotations = coco_data.annotations
        items = list(annotations)
        assert len(items) == 2

    def test_queryable_list_len(self, coco_data):
        """Test QueryableList supports len()."""
        annotations = coco_data.annotations
        assert len(annotations) == 2

    def test_queryable_list_indexing(self, coco_data):
        """Test QueryableList supports indexing."""
        annotations = coco_data.annotations
        first = annotations[0]
        assert isinstance(first, Annotation)


class TestAnnotation:
    """Test Annotation dataclass and torchvision conversion."""

    def test_annotation_to_dict_safe(self):
        """Test Annotation.to_dict_safe() preserves non-serializable objects."""
        ann = Annotation(
            id=1,
            image_id=1,
            category_id=1,
            bbox=[0, 0, 10, 10],
            area=100.0,
        )
        result = ann.to_dict_safe()
        assert isinstance(result, dict)
        assert result["id"] == 1
        assert result["bbox"] == [0, 0, 10, 10]

    def test_annotation_to_torchvision_bbox(self):
        """Test to_torchvision converts bbox to BoundingBoxes."""
        ann = Annotation(
            id=1,
            image_id=1,
            category_id=1,
            bbox=[100.0, 100.0, 50.0, 50.0],
            area=2500.0,
        )
        result = ann.to_torchvision(size=(480, 640))

        assert "bbox" in result
        bbox = result["bbox"]
        assert isinstance(bbox, torch.Tensor)
        assert bbox.shape == (1, 4)

    def test_annotation_to_torchvision_segmentation(self):
        """Test to_torchvision converts polygon segmentation to Mask."""
        ann = Annotation(
            id=2,
            image_id=1,
            category_id=2,
            segmentation=[[10, 10, 10, 20, 20, 20, 20, 10]],
            area=100.0,
        )
        result = ann.to_torchvision(size=(480, 640))

        assert "segmentation" in result

    def test_annotation_to_torchvision_rle_mask(self):
        """Test to_torchvision converts RLE mask."""
        ann = Annotation(
            id=3,
            image_id=1,
            category_id=1,
            mask={"size": [10, 10], "counts": [5, 2, 3, 10, 80]},
        )
        result = ann.to_torchvision(size=(10, 10))

        assert "mask" in result


class TestRLEListToMask:
    """Test RLE to mask conversion utility."""

    def test_basic(self):
        """Test rle_list_to_mask converts run-length encoding to binary mask."""
        rle = [5, 2, 3, 10, 80]  # 5 zeros, 2 ones, 3 zeros, 10 ones, 80 zeros
        mask = rle_list_to_mask(rle, height=10, width=10)

        assert mask.shape == (10, 10)
        assert mask.dtype == bool
        assert mask.sum() == 12  # 2 + 10 ones

    def test_all_zeros(self):
        """Test rle_list_to_mask with all zeros."""
        rle = [100]
        mask = rle_list_to_mask(rle, height=10, width=10)

        assert mask.shape == (10, 10)
        assert mask.sum() == 0

    def test_alternating(self):
        """Test rle_list_to_mask with alternating pattern."""
        rle = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        mask = rle_list_to_mask(rle, height=2, width=5)

        assert mask.shape == (2, 5)
        assert mask.sum() == 5


class TestCocoLabelerCanvasFallback:
    """``load_for`` must rasterize at the cube size when the COCO image record ships zero dims
    (some exporters leave height/width unset — e.g. the lentils day-level COCOs)."""

    @staticmethod
    def _zero_dim_coco(tmp_path):
        coco = {
            "info": {},
            "licenses": [{"id": 0, "name": "x"}],
            "categories": [{"id": 0, "name": "bg"}, {"id": 2, "name": "stone"}],
            "images": [{"id": 0, "file_name": "f.cu3s", "height": 0, "width": 0}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 0,
                    "category_id": 2,
                    "segmentation": [[2, 2, 8, 2, 8, 8, 2, 8]],
                    "area": 36.0,
                    "iscrowd": 0,
                }
            ],
        }
        p = tmp_path / "zerodim.json"
        p.write_text(json.dumps(coco))
        return p

    def test_falls_back_to_cube_size(self, tmp_path):
        import numpy as np

        from cuvis_ai_dataloader.data.labelers.coco_labeler import CocoLabeler

        lab = CocoLabeler(self._zero_dim_coco(tmp_path))
        out = lab.load_for(0, {"cube": np.zeros((10, 12, 1), dtype=np.float32)})["mask"]
        assert out.shape == (10, 12)  # cube size, NOT 0x0
        assert int(out.max()) == 2  # category rasterized at the real resolution
        assert int((out == 2).sum()) > 0

    def test_unannotated_image_zero_mask_at_cube_size(self, tmp_path):
        import numpy as np

        from cuvis_ai_dataloader.data.labelers.coco_labeler import CocoLabeler

        lab = CocoLabeler(self._zero_dim_coco(tmp_path))
        out = lab.load_for(99, {"cube": np.zeros((10, 12, 1), dtype=np.float32)})["mask"]
        assert out.shape == (10, 12) and int(out.max()) == 0


def _rect_mask(height, width, r0, r1, c0, c1):
    import numpy as np

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[r0:r1, c0:c1] = 1
    return mask


def _list_counts(mask):
    """Column-major alternating [bg, fg, ...] run lengths for a binary mask."""
    counts, prev, run = [], 0, 0
    for value in mask.flatten(order="F"):
        if int(value) == prev:
            run += 1
        else:
            counts.append(run)
            prev = int(value)
            run = 1
    counts.append(run)
    return counts


class TestTrackDialect:
    """CocoLabeler must read track-dialect files (top-level ``videos``, per-track parallel
    arrays, compressed-str RLE) — the shape the mask-tracking writer historically emitted —
    by converting them to image-keyed COCO in memory."""

    HEIGHT, WIDTH = 10, 12

    @classmethod
    def _payload(cls, with_optional=True):
        from cuvis_ai_core.data.rle import coco_rle_encode

        seg0 = coco_rle_encode(_rect_mask(cls.HEIGHT, cls.WIDTH, 2, 5, 3, 7))
        seg2 = coco_rle_encode(_rect_mask(cls.HEIGHT, cls.WIDTH, 6, 9, 1, 4))
        annotation = {
            "id": 1,
            "track_id": 7,
            "category_id": 2,
            "segmentations": [seg0, None, seg2],
        }
        if with_optional:
            annotation["detection_scores"] = [0.9, None, 0.8]
            annotation["bboxes"] = [[3.0, 2.0, 4.0, 3.0], None, [1.0, 6.0, 3.0, 3.0]]
            annotation["areas"] = [12.0, None, 9.0]
        return {
            "info": {"description": "Mask tracking results"},
            "videos": [
                {
                    "id": 1,
                    "name": "clip",
                    "frame_indices": [0, 1, 2],
                    "start_frame": 0,
                    "length": 3,
                    "height": cls.HEIGHT,
                    "width": cls.WIDTH,
                }
            ],
            "annotations": [annotation],
            "categories": [{"id": 2, "name": "stone"}],
        }

    @classmethod
    def _write(cls, tmp_path, payload):
        path = tmp_path / "track.json"
        path.write_text(json.dumps(payload))
        return path

    def _labeler(self, tmp_path, payload):
        from cuvis_ai_dataloader.data.labelers.coco_labeler import CocoLabeler

        return CocoLabeler(self._write(tmp_path, payload))

    def test_frames_become_image_ids(self, tmp_path):
        lab = self._labeler(tmp_path, self._payload())
        assert lab.image_ids == [0, 1, 2]
        assert lab.is_annotated(0) and lab.is_annotated(2)
        assert not lab.is_annotated(1)  # None-padded frame carries no annotation
        assert lab.categories_for(0) == [2]

    def test_load_for_rasterizes_track_masks(self, tmp_path):
        import numpy as np

        lab = self._labeler(tmp_path, self._payload())
        cube = np.zeros((self.HEIGHT, self.WIDTH, 1), dtype=np.float32)
        mask0 = lab.load_for(0, {"cube": cube})["mask"]
        assert mask0.shape == (self.HEIGHT, self.WIDTH) and mask0.dtype == np.int32
        assert (mask0[2:5, 3:7] == 2).all()
        assert int((mask0 == 2).sum()) == 12
        mask1 = lab.load_for(1, {"cube": cube})["mask"]
        assert int(mask1.max()) == 0
        mask2 = lab.load_for(2, {"cube": cube})["mask"]
        assert (mask2[6:9, 1:4] == 2).all()
        assert int((mask2 == 2).sum()) == 9

    def test_track_identity_and_scores_survive_conversion(self, tmp_path):
        lab = self._labeler(tmp_path, self._payload())
        raw_annotations = lab._coco._coco.dataset["annotations"]
        assert [ann["image_id"] for ann in raw_annotations] == [0, 2]
        assert all(ann["track_id"] == 7 for ann in raw_annotations)
        assert [ann["score"] for ann in raw_annotations] == [0.9, 0.8]
        assert all(ann["iscrowd"] == 1 for ann in raw_annotations)

    def test_bbox_and_area_derived_when_absent(self, tmp_path):
        lab = self._labeler(tmp_path, self._payload(with_optional=False))
        raw_annotations = lab._coco._coco.dataset["annotations"]
        first = raw_annotations[0]
        assert first["bbox"] == [3.0, 2.0, 4.0, 3.0]
        assert first["area"] == 12.0
        assert "score" not in first

    def test_hybrid_file_rejected(self, tmp_path):
        payload = self._payload()
        payload["images"] = [{"id": 0, "file_name": "f", "height": 10, "width": 12}]
        with pytest.raises(ValueError, match="hybrid"):
            self._labeler(tmp_path, payload)

    def test_empty_videos_rejected(self, tmp_path):
        payload = self._payload()
        payload["videos"] = []
        with pytest.raises(ValueError, match="empty or invalid"):
            self._labeler(tmp_path, payload)

    def test_multi_video_rejected(self, tmp_path):
        payload = self._payload()
        payload["videos"] = payload["videos"] + [dict(payload["videos"][0], id=2)]
        with pytest.raises(ValueError, match="unsupported"):
            self._labeler(tmp_path, payload)

    def test_duplicate_frame_indices_rejected(self, tmp_path):
        payload = self._payload()
        payload["videos"][0]["frame_indices"] = [0, 0, 2]
        with pytest.raises(ValueError, match="duplicates"):
            self._labeler(tmp_path, payload)

    def test_parallel_array_mismatch_rejected(self, tmp_path):
        payload = self._payload()
        payload["annotations"][0]["segmentations"] = payload["annotations"][0]["segmentations"][:2]
        with pytest.raises(ValueError, match="segmentations"):
            self._labeler(tmp_path, payload)

        payload = self._payload()
        payload["annotations"][0]["areas"] = [12.0, None]
        with pytest.raises(ValueError, match="areas"):
            self._labeler(tmp_path, payload)

    def test_non_rle_segmentation_rejected(self, tmp_path):
        payload = self._payload()
        payload["annotations"][0]["segmentations"][0] = [[1, 1, 5, 1, 5, 5, 1, 5]]
        with pytest.raises(ValueError, match="non-RLE"):
            self._labeler(tmp_path, payload)


class TestRleDictSegmentation:
    """Standard COCO RLE-object ``segmentation`` (str or list counts) must rasterize — the
    form image-dialect exports carry for masks — alongside the legacy ``mask`` key."""

    HEIGHT, WIDTH = 10, 12

    def _labeler_for_segmentation(self, tmp_path, segmentation, height=None, width=None):
        from cuvis_ai_dataloader.data.labelers.coco_labeler import CocoLabeler

        payload = {
            "info": {},
            "licenses": [{"id": 0, "name": "x"}],
            "images": [
                {
                    "id": 0,
                    "file_name": "frame_000000",
                    "height": height or self.HEIGHT,
                    "width": width or self.WIDTH,
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 0,
                    "category_id": 3,
                    "segmentation": segmentation,
                    "iscrowd": 1,
                }
            ],
            "categories": [{"id": 3, "name": "pill"}],
        }
        path = tmp_path / "image_rle.json"
        path.write_text(json.dumps(payload))
        return CocoLabeler(path)

    def _cube(self):
        import numpy as np

        return np.zeros((self.HEIGHT, self.WIDTH, 1), dtype=np.float32)

    def test_compressed_str_counts_rasterize(self, tmp_path):
        from cuvis_ai_core.data.rle import coco_rle_encode

        seg = coco_rle_encode(_rect_mask(self.HEIGHT, self.WIDTH, 2, 5, 3, 7))
        assert isinstance(seg["counts"], str)
        mask = self._labeler_for_segmentation(tmp_path, seg).load_for(0, {"cube": self._cube()})[
            "mask"
        ]
        assert (mask[2:5, 3:7] == 3).all()
        assert int((mask == 3).sum()) == 12

    def test_list_counts_rasterize(self, tmp_path):
        rect = _rect_mask(self.HEIGHT, self.WIDTH, 2, 5, 3, 7)
        seg = {"size": [self.HEIGHT, self.WIDTH], "counts": _list_counts(rect)}
        mask = self._labeler_for_segmentation(tmp_path, seg).load_for(0, {"cube": self._cube()})[
            "mask"
        ]
        assert (mask[2:5, 3:7] == 3).all()
        assert int((mask == 3).sum()) == 12

    def test_size_mismatch_fitted_to_canvas(self, tmp_path):
        from cuvis_ai_core.data.rle import coco_rle_encode

        seg = coco_rle_encode(_rect_mask(6, 6, 1, 4, 1, 4))  # declared 6x6, canvas 10x12
        mask = self._labeler_for_segmentation(tmp_path, seg).load_for(0, {"cube": self._cube()})[
            "mask"
        ]
        assert mask.shape == (self.HEIGHT, self.WIDTH)
        assert (mask[1:4, 1:4] == 3).all()
        assert int((mask == 3).sum()) == 9

    def test_bboxless_annotation_rasterizes(self, tmp_path):
        # No bbox key at all: rasterization must not depend on it.
        from cuvis_ai_core.data.rle import coco_rle_encode

        seg = coco_rle_encode(_rect_mask(self.HEIGHT, self.WIDTH, 0, 2, 0, 2))
        lab = self._labeler_for_segmentation(tmp_path, seg)
        mask = lab.load_for(0, {"cube": self._cube()})["mask"]
        assert int((mask == 3).sum()) == 4

    def test_to_torchvision_decodes_rle_dict(self):
        from cuvis_ai_core.data.rle import coco_rle_encode

        seg = coco_rle_encode(_rect_mask(self.HEIGHT, self.WIDTH, 2, 5, 3, 7))
        ann = Annotation(id=1, image_id=0, category_id=3, segmentation=seg)
        result = ann.to_torchvision(size=(self.HEIGHT, self.WIDTH))
        assert int(result["segmentation"].sum()) == 12


class TestSegmentationPayloadIntegrity:
    """``Annotation`` must carry the ``segmentation`` payload verbatim on every wizard version.

    The v1 loader (opt-in in dataclass-wizard 0.39.x, the default since 1.0) resolves a
    ``list | dict | None`` union list-first and coerces an RLE dict into the list of its
    keys, after which every RLE-object annotation rasterizes to 0 px (silent ground-truth
    loss on CuvisNEXT image-mode exports). The field is therefore typed ``Any``. These tests
    pin the passthrough by identity (the v0 union parser returned a copy, so a regression to
    typed parsing fails here even under the 0.39.x lock), exercise the v1 loader explicitly,
    and check that an unrecognized payload fails loudly instead of silently.
    """

    HEIGHT, WIDTH = 10, 12

    def _raw_ann(self, segmentation):
        return {
            "id": 8,
            "image_id": 0,
            "category_id": 3,
            "area": 12.0,
            "iscrowd": 1,
            "segmentation": segmentation,
        }

    def _rle(self):
        from cuvis_ai_core.data.rle import coco_rle_encode

        seg = coco_rle_encode(_rect_mask(self.HEIGHT, self.WIDTH, 2, 5, 3, 7))
        assert isinstance(seg["counts"], str)
        return seg

    def test_from_dict_passes_rle_dict_through_by_identity(self):
        seg = self._rle()
        ann = Annotation.from_dict(self._raw_ann(seg))
        assert ann.segmentation is seg

    def test_from_list_passes_rle_dict_through_by_identity(self):
        seg = self._rle()
        (ann,) = Annotation.from_list([self._raw_ann(seg)])
        assert ann.segmentation is seg

    def test_from_dict_passes_polygon_through_by_identity(self):
        poly = [[3.0, 2.0, 7.0, 2.0, 7.0, 5.0, 3.0, 5.0]]
        ann = Annotation.from_dict(self._raw_ann(poly))
        assert ann.segmentation is poly

    def test_v1_loader_preserves_rle_dict(self):
        """Opt a subclass into the v1 loader (the 1.x default) and check the RLE object survives.

        ``fromdict`` / ``fromlist`` are called directly instead of the inherited classmethods:
        once the base ``Annotation.from_dict`` has run, dataclass-wizard rebinds it to the
        generated v0 loader and a subclass defined afterwards inherits that binding, so the
        classmethod would exercise the wrong loader. The type checks guard that the subclass's
        own (v1) loader ran.
        """
        from dataclasses import dataclass

        from dataclass_wizard import JSONWizard, fromdict, fromlist

        @dataclass
        class V1Annotation(Annotation):
            class _(JSONWizard.Meta):
                v1 = True

        seg = self._rle()
        ann = fromdict(V1Annotation, self._raw_ann(seg))
        assert type(ann) is V1Annotation
        assert ann.segmentation is seg
        (listed,) = fromlist(V1Annotation, [self._raw_ann(seg)])
        assert type(listed) is V1Annotation
        assert listed.segmentation is seg

    def test_from_dict_rle_dict_rasterizes(self):
        from cuvis_ai_dataloader.data.labelers.coco_labeler import create_mask

        ann = Annotation.from_dict(self._raw_ann(self._rle()))
        mask = create_mask([ann], self.HEIGHT, self.WIDTH)
        assert int((mask == 3).sum()) == 12
        assert (mask[2:5, 3:7] == 3).all()

    def test_create_mask_raises_on_unrecognized_segmentation(self):
        from cuvis_ai_dataloader.data.labelers.coco_labeler import create_mask

        mangled = Annotation(id=8, image_id=0, category_id=3, segmentation=["counts", "size"])
        with pytest.raises(ValueError, match="unsupported 'segmentation' payload"):
            create_mask([mangled], self.HEIGHT, self.WIDTH)
