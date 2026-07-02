"""``cu3s-to-npz`` CLI: convert ``.cu3s`` session(s) into per-frame ``.npz`` for ``npz_multi``.

Reads each measurement (Preview -> Reflectance), bakes the frame's COCO annotations into
``mask`` + ``class_mask`` (when annotations are given), optionally crops, and writes one
``.npz`` per frame. **No train/val/test split is assigned** — splitting is a separate concern;
this only writes the npz (+ an optional traceability index ``npz_path,source_cu3s,image_id``).

Examples::

    cu3s-to-npz --cu3s-dir /data/lentils --out-dir /data/lentils_npz \
        --annotations sibling --index-csv /data/lentils_npz/index.csv
    cu3s-to-npz --cu3s a.cu3s b.cu3s --out-dir out --annotations coco.json --crop 300,300,300,300
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_crop(value: str) -> tuple[int, int, int, int]:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--crop must be TOP,BOTTOM,LEFT,RIGHT (4 ints)")
    try:
        t, b, left, r = (int(x) for x in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--crop values must be integers") from exc
    return (t, b, left, r)


def cu3s_to_npz_cli() -> None:
    """CLI entry point: convert cu3s session(s) to per-frame npz."""
    from cuvis_ai_dataloader.data.npz_converter import convert_cu3s

    parser = argparse.ArgumentParser(
        description="Convert .cu3s session(s) into per-frame .npz for the npz_multi loader. "
        "No split is assigned."
    )
    parser.add_argument("--cu3s", nargs="*", default=[], help="One or more .cu3s file paths.")
    parser.add_argument("--cu3s-dir", default=None, help="Directory to glob '**/*.cu3s' from.")
    parser.add_argument("--out-dir", required=True, help="Output directory for the .npz files.")
    parser.add_argument(
        "--annotations",
        default="sibling",
        help="COCO source: 'sibling' (<stem>.json next to each cu3s), 'none', "
        "or a path to one shared COCO json.",
    )
    parser.add_argument(
        "--crop",
        type=_parse_crop,
        default=None,
        metavar="T,B,L,R",
        help="Margins removed from each edge of cube+masks, e.g. 300,300,300,300.",
    )
    parser.add_argument(
        "--processing-mode",
        default="Reflectance",
        help="cuvis ProcessingMode (default Reflectance; 'none' uses the recorded cube).",
    )
    parser.add_argument(
        "--index-csv", default=None, help="Write a npz_path,source_cu3s,image_id index CSV here."
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Convert at most N frames per cu3s (0 = all)."
    )
    parser.add_argument("--no-compress", action="store_true", help="Write uncompressed .npz.")
    args = parser.parse_args()

    paths: list[Path] = [Path(p) for p in args.cu3s]
    if args.cu3s_dir:
        paths += sorted(Path(args.cu3s_dir).glob("**/*.cu3s"))
    if not paths:
        parser.error("provide --cu3s and/or --cu3s-dir")

    annotations = None if args.annotations == "none" else args.annotations
    processing_mode = None if str(args.processing_mode).lower() == "none" else args.processing_mode

    records = convert_cu3s(
        paths,
        args.out_dir,
        annotations=annotations,
        crop=args.crop,
        processing_mode=processing_mode,
        index_csv=args.index_csv,
        compress=not args.no_compress,
        frame_limit=args.limit or None,
    )
    print(f"wrote {len(records)} npz frame(s) from {len(paths)} cu3s into {args.out_dir}")
    if args.index_csv:
        print(f"index: {args.index_csv}")


if __name__ == "__main__":  # pragma: no cover
    cu3s_to_npz_cli()
