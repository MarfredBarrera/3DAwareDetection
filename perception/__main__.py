"""Verify one detection in one frame (run inside the container; check nvidia-smi first).

    sam3d-env core python -m perception --image frame.png --target chair --gpu 0
    sam3d-env core python -m perception --image frame.png --mask mask.png --target chair \\
        --selector top2gap --no-ensemble --elevation 20

Without ``--mask``, SAM3 segments the best ``--query`` (default: the target).
Writes input.png, mask.png, the rendered candidates and verdict.json to ``--out``.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from perception import select
from perception.client import HOUSEHOLD, Verifier


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--target", required=True, help="category the detector claimed")
    parser.add_argument("--mask", type=Path, help="object mask (nonzero = object)")
    parser.add_argument("--query", help="SAM3 prompt to find the object without --mask (default: target)")
    parser.add_argument("--elevation", type=float, default=0.0,
                        help="camera elevation in degrees, positive looking down")
    parser.add_argument("--selector", choices=select.SELECTORS, default="vqf")
    parser.add_argument("--no-ensemble", action="store_true", help="turn the ensemble filter off")
    parser.add_argument("--threshold", type=float, default=0.15, help="ensemble disagreement threshold")
    parser.add_argument("--on-uncertain", choices=select.ON_UNCERTAIN, default="input")
    parser.add_argument("--vocabulary", nargs="+", default=list(HOUSEHOLD))
    parser.add_argument("--all-views", action="store_true", help="render and classify every candidate")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/perception"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    with Verifier(selector=args.selector, ensemble=not args.no_ensemble, ensemble_threshold=args.threshold,
                  on_uncertain=args.on_uncertain, vocabulary=args.vocabulary, gpu=args.gpu,
                  all_views=args.all_views, log_dir=args.out / "logs") as verifier:
        mask = args.mask
        if mask is None:
            found = verifier.segment(args.image, args.query or args.target)
            if not found:
                raise SystemExit(f"SAM3 found no {args.query or args.target!r} in {args.image}")
            mask = found[0][0]
        verdict = verifier.verify(args.image, mask, args.target, args.elevation)
    from perception.images import to_pil
    to_pil(args.image).save(args.out / "input.png")
    to_pil(mask, "L").save(args.out / "mask.png")
    record = asdict(verdict)
    for view, data in zip(verdict.views, record["views"]):
        data["image"] = None
        if view.image is not None:
            data["image"] = f"view_a{view.azimuth:g}_e{view.elevation:g}.png"
            view.image.save(args.out / data["image"])
    if record["view"] is not None:
        record["view"]["image"] = f"view_a{verdict.view.azimuth:g}_e{verdict.view.elevation:g}.png"
    record.update(target=args.target, selector=args.selector, elevation=args.elevation)
    (args.out / "verdict.json").write_text(json.dumps(record, indent=1) + "\n")
    where = (f"SAM3D view {verdict.view.azimuth:g}/{verdict.view.elevation:g}" if verdict.judged == "view"
             else verdict.judged)
    uncertainty = "off" if verdict.uncertainty is None else f"{verdict.uncertainty:.3f}"
    print(f"{'ACCEPT' if verdict.accepted else 'REJECT'} {args.target!r}: top-1 {verdict.label!r} on {where} "
          f"(ensemble disagreement {uncertainty}) -> {args.out / 'verdict.json'}")


if __name__ == "__main__":
    main()
