"""Choose ViewActive objects: ImageNet-mappable LVIS classes, eligible meshes,
object-level splits stratified by class (the paper splits 70:5:25).

``python -m viewactive.cohort propose --top 200`` lists candidate ImageNet
classes for unmapped LVIS categories, to extend the reviewed class map.
"""
import argparse
import gzip
import json
import math
import random
from pathlib import Path

SPLITS = ("train", "val", "test")
UNSEEN = "test_unseen"


def load_class_map(path):
    """LVIS category -> sorted ImageNet-1k indices; ``"a-b"`` keys are inclusive ranges."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path} must map LVIS categories to ImageNet classes")
    mapping = {}
    for category, classes in raw.items():
        if not isinstance(classes, dict) or not classes:
            raise ValueError(f"{path}: {category} must map ImageNet indices to descriptions")
        indices = set()
        for key in classes:
            first, _, last = str(key).partition("-")
            first, last = int(first), int(last or first)
            if not 0 <= first <= last < 1000:
                raise ValueError(f"{path}: {category} has an invalid ImageNet index {key}")
            indices.update(range(first, last + 1))
        mapping[category] = sorted(indices)
    return mapping


def lvis_categories(nas_root):
    with gzip.open(Path(nas_root) / "hf-objaverse-v1" / "lvis-annotations.json.gz", "rb") as f:
        return json.load(f)


def object_paths(nas_root):
    with gzip.open(Path(nas_root) / "hf-objaverse-v1" / "object-paths.json.gz", "rb") as f:
        return json.load(f)


def excluded_uids(paths):
    """Objects reserved for the viewpoint study, e.g. its reviewed canonical.json."""
    return {uid for path in paths for uid in json.loads(Path(path).read_text())["objects"]}


def choose_classes(cohort, lvis, class_map, excluded=()):
    """(seen, unseen) category lists; counts rank mappable classes by object count."""
    available = {c: len(set(lvis[c]) - set(excluded)) for c in class_map if c in lvis}
    missing = set(class_map) - set(lvis)
    if missing:
        raise ValueError(f"Class map categories absent from LVIS: {sorted(missing)}")
    ranking = sorted(available, key=lambda c: (-available[c], c))
    chosen = {}
    for key in ("seen_classes", "unseen_classes"):
        value = cohort[key]
        if isinstance(value, list):
            unknown = set(value) - set(available)
            if unknown:
                raise ValueError(f"cohort.{key} lists unmapped categories: {sorted(unknown)}")
            chosen[key] = list(value)
    taken = {c for classes in chosen.values() for c in classes}
    for key in ("seen_classes", "unseen_classes"):
        if key not in chosen:
            pool = [c for c in ranking if c not in taken]
            if len(pool) < cohort[key]:
                raise ValueError(f"Only {len(pool)} mappable classes remain for cohort.{key}")
            chosen[key] = pool[:cohort[key]]
            taken.update(chosen[key])
    return chosen["seen_classes"], chosen["unseen_classes"]


def candidates(cohort, lvis, seen, unseen, excluded=(), local=()):
    """Ordered (uid, category) pairs: a seeded shuffle per class, locally stored
    GLBs first, ``oversample`` x the needed count to absorb unusable meshes."""
    rng = random.Random(cohort["seed"])
    excluded, local, used = set(excluded), set(local), set()
    pairs = []
    for classes, count in ((seen, cohort["per_class"]), (unseen, cohort["unseen_per_class"])):
        for category in classes:
            pool = sorted(set(lvis[category]) - excluded - used)
            rng.shuffle(pool)
            pool.sort(key=lambda uid: uid not in local)
            chosen = pool[:math.ceil(count * cohort["oversample"])]
            used.update(chosen)
            pairs.extend((uid, category) for uid in chosen)
    return pairs


def eligible(record, max_faces, drop_suspect=False):
    return record["face_count"] <= max_faces and not (drop_suspect and record["up_axis_suspect"])


def _apportion(counts, fraction, target, capacity=None):
    """Allocate a global split total across classes by largest remainder."""
    capacity = counts if capacity is None else capacity
    quotas = {category: min(math.floor(count * fraction), capacity[category])
              for category, count in counts.items()}
    remaining = target - sum(quotas.values())
    # Sort fractional remainders descending, with names as deterministic ties.
    order = sorted(counts, key=lambda category: (-(counts[category] * fraction
                                                   - math.floor(counts[category] * fraction)), category))
    for category in order:
        if remaining <= 0:
            break
        if quotas[category] < capacity[category]:
            quotas[category] += 1
            remaining -= 1
    if remaining:
        raise ValueError("Cannot allocate the requested split across classes")
    return quotas


def finalize(cohort, pairs, canonical, seen, unseen, max_faces, drop_suspect=False):
    """{uid: {"category", "split"}} from the first eligible candidates per class.

    ``drop_suspect`` removes objects whose up axis ``harness.canonicalize`` flagged."""
    rng = random.Random(cohort["seed"])
    members, short = {}, {}
    total = cohort["total_seen"]
    for classes, count, require_full in ((seen, cohort["per_class"], total is None),
                                         (unseen, cohort["unseen_per_class"], True)):
        for category in classes:
            usable = [uid for uid, c in pairs
                      if c == category and uid in canonical
                      and eligible(canonical[uid], max_faces, drop_suspect)]
            # A capped seen cohort can absorb a short class while retaining
            # its category, provided the other classes cover the total.
            if len(usable) < count and (require_full or not usable):
                short[category] = len(usable)
            members[category] = usable[:count]
    if short:
        raise ValueError("Too few eligible objects (raise cohort.oversample or drop the class): "
                         + ", ".join(f"{c}={n}" for c, n in sorted(short.items())))
    if total is not None:
        if total > sum(len(members[c]) for c in seen):
            raise ValueError("cohort.total_seen exceeds seen_classes x per_class")
        # Trim one object at a time from the least common classes first.
        order = list(reversed(seen))
        while sum(len(members[c]) for c in seen) > total:
            category = order.pop(0)
            members[category].pop()
            order.append(category)
    counts = {category: len(members[category]) for category in seen}
    total_seen = sum(counts.values())
    n_test = round(total_seen * cohort["split"][2])
    n_val = round(total_seen * cohort["split"][1])
    test_quota = _apportion(counts, cohort["split"][2], n_test)
    val_quota = _apportion(counts, cohort["split"][1], n_val,
                           {category: counts[category] - test_quota[category] for category in seen})
    objects = {}
    for category in seen:
        uids = sorted(members[category])
        rng.shuffle(uids)
        for i, uid in enumerate(uids):
            split = ("test" if i < test_quota[category] else
                     "val" if i < test_quota[category] + val_quota[category] else "train")
            objects[uid] = {"category": category, "split": split}
    for category in unseen:
        for uid in members[category]:
            objects[uid] = {"category": category, "split": UNSEEN}
    return dict(sorted(objects.items()))


def propose(nas_root, class_map_path, top):
    """Print token-overlap ImageNet candidates for the most common unmapped categories."""
    from timm.data import ImageNetInfo

    info = ImageNetInfo()
    names = {i: info.index_to_description(i).lower() for i in range(1000)}
    mapped = set(load_class_map(class_map_path)) if Path(class_map_path).is_file() else set()
    lvis = lvis_categories(nas_root)
    for category in sorted(lvis, key=lambda c: -len(lvis[c]))[:top]:
        if category in mapped:
            continue
        tokens = {t for t in category.lower().replace("(", " ").replace(")", " ").replace("_", " ").split()
                  if len(t) > 2}
        hits = [f"{i}: {names[i].split(',')[0]}" for i in names
                if tokens & set(names[i].replace(",", " ").replace("-", " ").split())]
        print(f"{category} ({len(lvis[category])}): {'; '.join(hits[:8]) or '-'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    proposal = sub.add_parser("propose")
    proposal.add_argument("--nas-root", default="/data/objaverse_nas")
    proposal.add_argument("--class-map", default="configs/viewactive_imagenet_map.yaml")
    proposal.add_argument("--top", type=int, default=200)
    args = parser.parse_args()
    propose(args.nas_root, args.class_map, args.top)


if __name__ == "__main__":
    main()
