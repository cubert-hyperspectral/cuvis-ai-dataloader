"""``cu3s-to-npz`` CLI: convert ``.cu3s`` session(s) into per-frame ``.npz`` for ``npz_multi``.

Reads each measurement (Preview -> Reflectance), bakes the frame's COCO annotations into
``mask`` + ``class_mask`` (when annotations are given), optionally crops, and writes one
``.npz`` per frame. **No train/val/test split is assigned** — splitting is a separate concern;
this only writes the npz (+ an optional universe ``source,index,materialized_path``).

Examples::

    cu3s-to-npz --cu3s-dir /data/lentils --out-dir /data/lentils_npz \
        --annotations sibling --universe-csv /data/lentils_npz/universe.csv
    cu3s-to-npz --cu3s a.cu3s b.cu3s --out-dir out --annotations coco.json --crop 300,300,300,300

    # Re-reference a factory-fallback session against day-matched white/dark recordings:
    cu3s-to-npz --cu3s bad_calib.cu3s --out-dir out --annotations none \
        --white-ref day2_white.cu3s --dark-ref day2_dark.cu3s
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
        "--white-ref",
        default=None,
        metavar="CU3S[:FRAME]",
        help="cu3s reference supplied as the White reference (overriding the baked one): "
        "'path'/'path:0' = measurement 0, 'path:N' = measurement N, 'path:-1' = the session's "
        "embedded reference. Applies to every input; use references matching the capture conditions.",
    )
    parser.add_argument(
        "--dark-ref",
        default=None,
        metavar="CU3S[:FRAME]",
        help="cu3s reference supplied as the Dark reference (overriding the baked one): "
        "'path'/'path:0' = measurement 0, 'path:N' = measurement N, 'path:-1' = the session's "
        "embedded reference. Applies to every input; use references matching the capture conditions.",
    )
    parser.add_argument(
        "--universe-csv",
        default=None,
        help="Write a source,index,materialized_path universe CSV here.",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Convert at most N frames per cu3s (0 = all)."
    )
    parser.add_argument("--no-compress", action="store_true", help="Write uncompressed .npz.")
    parser.add_argument(
        "--read-threads",
        type=int,
        default=0,
        help="Read each cu3s on N session handles sharing one processing context (0 = off, "
        "6 recommended on GPU). Needs a cuvis binding that releases the GIL; on one that does "
        "not it warns and reads single-threaded.",
    )
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
        white_ref=args.white_ref,
        dark_ref=args.dark_ref,
        universe_csv=args.universe_csv,
        compress=not args.no_compress,
        frame_limit=args.limit or None,
        read_threads=args.read_threads,
    )
    print(f"wrote {len(records)} npz frame(s) from {len(paths)} cu3s into {args.out_dir}")
    if args.universe_csv:
        print(f"universe: {args.universe_csv}")


if __name__ == "__main__":  # pragma: no cover
    cu3s_to_npz_cli()
