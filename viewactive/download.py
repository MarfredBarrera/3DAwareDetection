"""Fetch candidate GLBs and write an ``ObjaverseStore``-compatible root per run.

GLBs on the read-only NAS slice are used in place; others are downloaded from
``allenai/objaverse`` into ``cache_root`` (shared between runs). The run's store
maps every uid to an absolute GLB path, so ``harness.canonicalize`` and
``harness.meshio.ObjaverseStore`` work unchanged (``os.path.join`` keeps an
absolute second argument).
"""
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path

REPO = "allenai/objaverse"


def ensure_glbs(uids, paths, *, nas_root, cache_root, max_mb, workers=8, log=print):
    """Return ({uid: absolute GLB path}, {uid: skip reason})."""
    from huggingface_hub import HfApi, hf_hub_download

    limit = max_mb * 2**20
    found, skipped, missing = {}, {}, []
    for uid in uids:
        relative = paths.get(uid)
        if relative is None:
            skipped[uid] = "no_object_path"
            continue
        for root in (Path(nas_root) / "hf-objaverse-v1", Path(cache_root)):
            if (root / relative).is_file():
                found[uid] = root / relative
                break
        else:
            missing.append(uid)
    sizes = {}
    api = HfApi()
    for start in range(0, len(missing), 100):
        chunk = [paths[uid] for uid in missing[start:start + 100]]
        sizes.update({info.path: info.size for info in api.get_paths_info(REPO, chunk, repo_type="dataset")})
    wanted = []
    for uid in missing:
        size = sizes.get(paths[uid])
        if size is None:
            skipped[uid] = "not_on_hub"
        elif size > limit:
            skipped[uid] = "oversize"
        else:
            wanted.append(uid)
    log(f"[download] {len(found)} local, {len(wanted)} to download, {len(skipped)} skipped")

    def fetch(uid):
        try:
            return uid, Path(hf_hub_download(REPO, paths[uid], repo_type="dataset",
                                             local_dir=cache_root)), None
        except Exception as error:  # Keep going; the cohort stage absorbs losses.
            return uid, None, f"download_failed: {type(error).__name__}"

    with ThreadPoolExecutor(workers) as pool:
        for n, (uid, path, error) in enumerate(pool.map(fetch, wanted), 1):
            if path is None:
                skipped[uid] = error
            else:
                found[uid] = path
            if n % 100 == 0:
                log(f"[download] {n}/{len(wanted)}")
    for uid, path in list(found.items()):
        if path.stat().st_size > limit:
            skipped[uid] = "oversize"
            del found[uid]
    return {uid: str(path) for uid, path in found.items()}, skipped


def write_store(store_root, glbs, categories):
    """Write manifest.json and object-paths.json.gz for ``harness`` tools."""
    version = Path(store_root) / "hf-objaverse-v1"
    version.mkdir(parents=True, exist_ok=True)
    manifest = {"uid_to_categories": {uid: [categories[uid]] for uid in sorted(glbs)}}
    (Path(store_root) / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    with gzip.open(version / "object-paths.json.gz", "wt") as stream:
        json.dump(dict(sorted(glbs.items())), stream)
