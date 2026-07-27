# TODOs

## npz_multi could supply category_ids from its class_mask

**What:** `MultiNpzDataModule.enumerate` currently raises `NotImplementedError` when
`tags` / `category_ids` are requested and declares `supported_attrs() == frozenset()`. It
could instead derive per-frame anomaly metadata from the baked NPZ `class_mask` (a frame is
anomalous iff its mask contains a label id != 0), populating `category_ids` on each
`SampleRef`.

**Why:** the `no_train_anomalous` split constraint currently evaluates to `unavailable`
(soft-skip) on the npz path, because npz can't report anomaly labels. If npz supplied them,
core could *verify* normal-only-ness of an npz training split instead of skipping the check.

**Current state:** the npz dinomaly split is normal-only by construction (anomalous frames
live in a separate adaclip pool), so the soft-skip is honest and this is not blocking. Only
needed if we want backend enforcement of the anomaly constraint on an npz split.

**Where to start:** `cuvis_ai_dataloader/data/datamodule_npz_multi.py::enumerate` (read the
`class_mask` array per row, set `category_ids=[sorted non-zero label ids]`) and flip
`supported_attrs()` to include `"category_ids"`.

**Depends on:** the constraints release train (schemas + core with the constraint evaluator).
