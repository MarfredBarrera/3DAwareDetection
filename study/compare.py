"""Draw an input-versus-optimized viewpoint grid for one study object.

An optional picks JSON maps method -> input image key -> candidate image key,
for example {"viewactive": {"gt/<uid>/a0_e0": "sam3d/..."}}. Every candidate
must belong to that input's reconstruction grid. This uses cached images and
SAM3 scores; it runs no model inference.
"""
import argparse
import json
import math
from pathlib import Path

from study import config as settings_io
from study import db as store
from study.figures import save_view_comparison
from study.labels import outcomes


def build(config, uid, selector, path, *, picks=None):
    """Save the same four-panel PNG for any stored or externally supplied selector."""
    output = Path(config["output_dir"])
    if picks is not None and not isinstance(picks, dict):
        raise ValueError("picks must map method names to input-key/candidate-key mappings")
    db = store.connect(output)
    try:
        members = store.images(db, "i.source='gt' AND i.uid=?", (uid,))
        if not members:
            raise ValueError(f"Unknown object {uid!r}")
        rows = {row["input_id"]: row for row in db.execute(
            "SELECT * FROM selections WHERE selector=?", (selector,))}
        external = None if picks is None else picks.get(selector)
        if not rows and external is None:
            raise ValueError(f"Unknown selector {selector!r}; run analyze or supply --picks")
        if external is not None and not isinstance(external, dict):
            raise ValueError(f"picks for {selector!r} must map input keys to candidate keys")

        ids = [row["image_id"] for row in members]
        placeholders = ",".join("?" for _ in ids)
        vocabulary = config["dataset"]["categories"]
        detection = config["detection"]
        before_outcomes = outcomes(db, vocabulary, detection,
                                   f"i.image_id IN ({placeholders})", ids)
        candidate_rows = store.images(db, f"i.parent_id IN ({placeholders})", ids)
        candidates = {row["image_id"]: row for row in candidate_rows}
        by_key = {row["key"]: row for row in candidate_rows}
        chosen = {}
        for source in members:
            input_id = source["image_id"]
            if external is not None:
                key = external.get(source["key"])
                if key is None:
                    continue
                pick = by_key.get(key)
                if pick is None or pick["parent_id"] != input_id:
                    raise ValueError(
                        f"{selector} picked {key!r}, which is not a candidate of {source['key']!r}")
            else:
                selection = rows.get(input_id)
                if selection is None:
                    continue
                pick = candidates.get(selection["pick_id"])
                if pick is None or pick["parent_id"] != input_id:
                    raise ValueError(f"{selector} has no candidate for {source['key']!r}")
            chosen[input_id] = pick
        if not chosen:
            raise ValueError(f"No {selector} picks found for object {uid!r}")

        candidate_outcomes = outcomes(db, vocabulary, detection,
                                      f"i.parent_id IN ({placeholders})", ids)
        angle = lambda row: (row["azimuth"], row["elevation"])
        before = {angle(source): before_outcomes.get(source["image_id"], {}).get("margin")
                  for source in members}
        paths = {angle(source): output / chosen[source["image_id"]]["path"]
                 for source in members if source["image_id"] in chosen}
        after = {angle(source): candidate_outcomes.get(chosen[source["image_id"]]["image_id"], {}).get("margin")
                 for source in members if source["image_id"] in chosen}
        all_margins = [result["margin"] for result in (*before_outcomes.values(),
                                                       *candidate_outcomes.values())
                       if math.isfinite(result["margin"])]
        limit = max((abs(value) for value in all_margins), default=1.0) or 1.0
        return save_view_comparison(config, members[0]["category"], uid, members, before,
                                    paths, after, selector, path, limit=limit)
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--uid", help="object UID from the study")
    target.add_argument("--all", action="store_true", help="every object with analyzed inputs")
    parser.add_argument("--selector", help="stored selector or method name in --picks")
    parser.add_argument("--selectors", nargs="+", help="methods to plot with --all")
    parser.add_argument("--picks", type=Path, help="JSON mapping method -> input key -> candidate key")
    parser.add_argument("--out", type=Path, help="output PNG; defaults to figures/comparison/<selector>/<uid>.png")
    args = parser.parse_args()
    config = settings_io.load(args.config)
    picks = json.loads(args.picks.read_text()) if args.picks else None
    if args.all:
        if args.out is not None or args.selector is not None:
            parser.error("--all uses --selectors and the standard output directories")
        selectors = args.selectors or config["analysis"].get("figure_selectors") or [
            config["analysis"]["primary_selector"]]
        with store.connect(config["output_dir"]) as db:
            uids = [row[0] for row in db.execute(
                "SELECT DISTINCT i.uid FROM selections s JOIN images i ON i.image_id=s.input_id"
                " ORDER BY i.uid")]
        for selector in selectors:
            for uid in uids:
                path = (Path(config["output_dir"]) / "figures" / "comparison"
                        / selector / f"{uid}.png")
                build(config, uid, selector, path, picks=picks)
        print(f"Saved {len(uids) * len(selectors)} comparison grids for {len(uids)} objects")
    else:
        if args.selectors is not None:
            parser.error("--selectors requires --all")
        selector = args.selector or config["analysis"]["primary_selector"]
        path = (args.out or Path(config["output_dir"]) / "figures" / "comparison"
                / selector / f"{args.uid}.png")
        print(build(config, args.uid, selector, path, picks=picks))


if __name__ == "__main__":
    main()
