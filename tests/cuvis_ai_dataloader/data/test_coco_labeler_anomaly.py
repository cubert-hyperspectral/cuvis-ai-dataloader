"""Tests for the CocoLabeler anomaly predicate (AD: train = normal frames only)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


def _ann(category_id, segmentation=None, mask=None):
    return SimpleNamespace(category_id=category_id, segmentation=segmentation, mask=mask)


def _labeler_with(annotations_by_image):
    """Build a CocoLabeler over a mocked COCOData with the given annotations."""
    mock_coco = Mock()
    mock_coco.category_id_to_name = {0: "background", 1: "stone", 2: "fly"}
    mock_coco.image_ids = sorted(annotations_by_image)
    mock_coco.annotations = Mock()
    mock_coco.annotations.where = Mock(
        side_effect=lambda image_id: annotations_by_image.get(image_id, [])
    )
    with patch(
        "cuvis_ai_dataloader.data.labelers.coco_labeler.COCOData.from_path",
        return_value=mock_coco,
    ):
        from cuvis_ai_dataloader.data.labelers.coco_labeler import CocoLabeler

        return CocoLabeler(annotation_json_path="fake.json")


TRIANGLE = [[0.0, 0.0, 10.0, 0.0, 10.0, 10.0]]  # 3 points = rasterizable polygon


@pytest.mark.parametrize(
    ("annotations", "expected_anomalous", "expected_cats"),
    [
        # real polygon, non-normal category -> anomalous
        ([_ann(1, segmentation=TRIANGLE)], True, {1}),
        # real RLE, non-normal category -> anomalous
        ([_ann(2, mask={"counts": [5, 3, 2], "size": [8, 8]})], True, {2}),
        # multiple effective categories union
        (
            [_ann(1, segmentation=TRIANGLE), _ann(2, mask={"counts": [1], "size": [8, 8]})],
            True,
            {1, 2},
        ),
        # category 0 never counts, even with a real polygon
        ([_ann(0, segmentation=TRIANGLE)], False, set()),
        # placeholder annotations (the exported-lentils case): no pixels -> normal
        ([_ann(1, mask={"counts": [], "size": [8, 8]})], False, set()),
        ([_ann(1, segmentation=[])], False, set()),
        ([_ann(1, segmentation=[[0.0, 0.0, 1.0, 1.0]])], False, set()),  # <3 points
        ([_ann(1)], False, set()),
        # no annotations at all -> normal
        ([], False, set()),
    ],
)
def test_is_anomalous_matches_rasterization(annotations, expected_anomalous, expected_cats):
    labeler = _labeler_with({0: annotations})
    assert labeler.is_anomalous(0) is expected_anomalous
    assert labeler.anomaly_category_ids(0) == frozenset(expected_cats)


def test_frame_absent_from_coco_is_normal():
    labeler = _labeler_with({0: [_ann(1, segmentation=TRIANGLE)]})
    assert labeler.is_anomalous(99) is False
    assert labeler.anomaly_category_ids(99) == frozenset()


def test_mixed_frames_classify_independently():
    labeler = _labeler_with(
        {
            0: [],
            1: [_ann(1, segmentation=TRIANGLE)],
            2: [_ann(0, segmentation=TRIANGLE)],
        }
    )
    assert [labeler.is_anomalous(i) for i in (0, 1, 2)] == [False, True, False]
