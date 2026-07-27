"""``resolve-splits`` CLI: generate a committable ``splits.json`` for a data module.

Two modes:

* ``--from-csv lentils_splits.csv --out splits.json``: import a cu3s_multi CSV's ``split``
  column into per-source FILE_INDICES selectors (outcome-equivalent to the CSV).
* ``--data-module cu3s --data-arg ... --strategy stratified --seed 7 --out splits.json``:
  enumerate the module's universe and split it (random / stratified, optional ``--ad-aware``
  train-on-normals and ``--group-by source`` to keep a file whole).
"""

from __future__ import annotations

import argparse


def _build_module(data_module: str, data_args: dict[str, str]):
    from cuvis_ai_dataloader.data.datamodule_cu3s import Cu3sDataModule
    from cuvis_ai_dataloader.data.datamodule_cu3s_multi import MultiCu3sDataModule
    from cuvis_ai_dataloader.data.datamodule_tiff_paired import TiffPairedDataModule

    classes = {
        "cu3s": Cu3sDataModule,
        "tiff_paired": TiffPairedDataModule,
        "cu3s_multi": MultiCu3sDataModule,
    }
    cls = classes.get(data_module)
    if cls is None:
        raise SystemExit(
            f"--data-module {data_module!r} not provided by cuvis-ai-dataloader "
            f"(have: {sorted(classes)})"
        )
    cls.validate_params(data_args)
    return cls(params=data_args)


def resolve_splits_cli() -> None:
    """CLI entry point: write a ``splits.json`` for a data module."""
    from cuvis_ai_core.data.splits_io import save_splits

    from cuvis_ai_dataloader.data.resolvers import (
        import_csv_splits,
        resolve_random,
        resolve_stratified,
    )

    parser = argparse.ArgumentParser(
        description="Generate a committable splits.json for a cuvis-ai data module.",
    )
    parser.add_argument(
        "--from-csv", default=None, help="Import a cu3s_multi universe.csv's 'split' column."
    )
    parser.add_argument(
        "--data-module", default=None, help="Data module name to enumerate + split."
    )
    parser.add_argument(
        "--data-arg",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Module-specific argument (repeatable), e.g. --data-arg cu3s_file_path=X.cu3s.",
    )
    parser.add_argument("--strategy", choices=["random", "stratified"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--group-by", choices=["source", "group"], default=None)
    parser.add_argument("--ad-aware", action="store_true", help="Train on normals only.")
    parser.add_argument("--out", required=True, help="Output splits.json path.")
    args = parser.parse_args()

    if args.from_csv:
        from cuvis_ai_dataloader.data.datamodule_cu3s_multi import MultiCu3sDataModule

        module = MultiCu3sDataModule(params={"universe_csv": args.from_csv})
        splits = import_csv_splits(module)
    else:
        if not args.data_module:
            parser.error("provide either --from-csv or --data-module")
        data_args: dict[str, str] = {}
        for pair in args.data_arg or []:
            if "=" not in pair:
                parser.error(f"--data-arg must be KEY=VALUE, got {pair!r}")
            key, value = pair.split("=", 1)
            data_args[key.strip()] = value
        module = _build_module(args.data_module, data_args)
        need_attrs = args.ad_aware or args.strategy == "stratified"
        refs = module.enumerate(frozenset({"category_ids"}) if need_attrs else frozenset())
        resolver = resolve_stratified if args.strategy == "stratified" else resolve_random
        splits = resolver(
            refs,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            group_by=args.group_by,
            ad_aware=args.ad_aware,
        )

    save_splits(splits, args.out)
    print(
        f"wrote {args.out}: train={len(splits.train)} val={len(splits.val)} "
        f"test={len(splits.test)} selectors"
    )


if __name__ == "__main__":  # pragma: no cover
    resolve_splits_cli()
