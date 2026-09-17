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


## `samples_per_frame > 1` loses the batched fetch (core `_RepeatDataset`)

**What:** `cuvis_ai_core.data.datamodule._RepeatDataset` defines only `__len__` and
`__getitem__`, so torch's fetcher sees no `__getitems__` and falls back to per-index reads on
the train loader whenever `samples_per_frame > 1`. Val, test and predict are unaffected.

**Why:** `read_threads` only overlaps reads inside one `__getitems__` call; a repeated train
loader reads single-threaded however many threads are configured.

**Where to start:** a three-line forward in cuvis-ai-core: `def __getitems__(self, indices):
return self._base.__getitems__([i % len(self._base) for i in indices])` when the base has one.
Then drop the "train loader only" caveat from the README here.

**Depends on:** a cuvis-ai-core release; nothing in this repository.

## Sdist hygiene beyond `benchmarks/`

**What:** the sdist ships every git-tracked file (setuptools-scm's file finder, no
`MANIFEST.in` until 0.7.0 pruned `benchmarks/`): `tests/`, `.github/`, `.githooks/`,
`.claude/`, `CLAUDE.md`, `AGENTS.md`, `TODOS.md`, `uv.lock`.

**Why:** none of it is needed to install or use the package, and a binary committed anywhere in
the tree lands on PyPI, which cannot be republished.

**Where to start:** extend `MANIFEST.in` with `prune` lines for the directories above and
`exclude` for the root files, then check `uv build --sdist` and the "Build and validate
package" job.

**Depends on:** nothing.

## Release job: import the cu3s binding from the built wheel

**What:** `.github/workflows/pypi-release.yml` validates the built artifacts' metadata inside the
`cuvis_pyil` image but never installs the wheel with the `cu3s` extra and imports `cuvis`.

**Why:** the `cu3s` extra's floor moved to `cuvis>=3.6.0.0`, whose binding fails at import
against a 3.5.x native SDK; a mismatch between the image and the binding would only surface at a
user's first read.

**Where to start:** after the build step, `uv pip install dist/*.whl[cu3s]` into a scratch env
inside the image and run `python -c "import cuvis; print(cuvis.version())"`.

**Depends on:** nothing.
