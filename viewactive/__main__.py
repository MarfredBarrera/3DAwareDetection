"""Reproduce ViewActive (Wu et al., arXiv:2409.09997) end to end.

Run inside the container (check nvidia-smi and set gpu_id first)::

    sam3d-env sam3d python -m viewactive --config configs/viewactive_smoke.yaml
    sam3d-env sam3d python -m viewactive --config configs/viewactive.yaml --stages train evaluate

Stages: select (LVIS candidates) -> download (GLBs) -> canonicalize -> cohort
(eligible objects, splits) -> render (12 x 11 grid, verified cameras) -> label
(VQF) -> features (frozen DINOv2 + CLIP) -> classify (ImageNet top-1) ->
train -> evaluate. Data stages fill in what is missing, so rerunning resumes;
changing their settings requires a new output_dir. ``train`` reruns when its
settings change; ``evaluate`` always reruns. Logs go to <output_dir>/logs/.
"""
import argparse
import fcntl
import json
import multiprocessing
from pathlib import Path

import numpy as np

from study import config as settings_io
from tools.envs import no_huge_pages, run_in_env, use_gpu
from viewactive import cohort as cohorts
from viewactive.grid import Grid

STAGES = ("select", "download", "canonicalize", "cohort", "render", "label", "features",
          "classify", "train", "evaluate")
FINGERPRINT = {"select": "cohort", "download": "cohort", "canonicalize": "cohort", "cohort": "cohort",
               "render": "render", "label": "label", "features": "features", "classify": "classify"}


class Run:
    """Paths and shared reads for one output directory."""

    def __init__(self, config):
        self.config, self.dir = config, Path(config["output_dir"])
        self.grid = Grid.from_config(config["render"])
        self.store = self.dir / "store"
        self.canonical = self.dir / "canonical.json"
        self.vqf = self.dir / "vqf.npz"
        self.checkpoint = self.dir / "model" / "best.pt"

    def read(self, name):
        return json.loads((self.dir / name).read_text())

    def write(self, name, value):
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=1) + "\n")

    def objects(self):
        return self.read("objects.json")

    def shards(self):
        uids = sorted(self.objects())
        size = self.config["render"]["shard_size"]
        return [uids[i:i + size] for i in range(0, len(uids), size)]

    def shard_dir(self, number):
        return self.dir / "renders" / f"shard-{number:03d}"

    def views(self):
        """uid -> [E][A] (rgb path, modal mask path) from every shard index."""
        rows = {el: i for i, el in enumerate(self.grid.elevations)}
        cols = {az: j for j, az in enumerate(self.grid.azimuths)}
        out = {}
        for number in range(len(self.shards())):
            directory = self.shard_dir(number)
            for line in (directory / "index.jsonl").read_text().splitlines():
                row = json.loads(line)
                cells = out.setdefault(row["uid"], [[None] * len(cols) for _ in rows])
                cells[rows[row["elevation"]]][cols[row["azimuth"]]] = (
                    str(directory / row["files"]["rgb"]), str(directory / row["files"]["modal"]))
        return out

    def labelled(self):
        with np.load(self.vqf) as data:
            return {key: data[key] for key in data.files}

    def features_path(self):
        from viewactive.features import encoder_id
        encoders = self.config["encoders"]
        return self.dir / "features" / f"{encoder_id(encoders['dino'], encoders['clip'])}.npy"

    def check(self, stage, strict=True, record=False):
        """Record or compare a stage fingerprint; False if it changed and ``strict`` is off.
        ``record`` overwrites the stored value (after recomputing a stage)."""
        path = self.dir / "fingerprints.json"
        stored = json.loads(path.read_text()) if path.is_file() else {}
        value = settings_io.fingerprint(self.config, stage, settings_io.VIEWACTIVE_FINGERPRINTS)
        if not record and stored.get(stage) not in (None, value):
            if strict:
                raise ValueError(f"{stage} settings changed since {self.dir} was created; "
                                 "use a new output_dir")
            return False
        path.write_text(json.dumps(stored | {stage: value}, indent=1) + "\n")
        return True


def select(run, log):
    if (run.dir / "candidates.json").is_file():
        return
    config = run.config
    lvis = cohorts.lvis_categories(config["objaverse"]["nas_root"])
    class_map = cohorts.load_class_map(config["cohort"]["class_map"])
    excluded = cohorts.excluded_uids(config["cohort"]["exclude_uids_from"])
    seen, unseen = cohorts.choose_classes(config["cohort"], lvis, class_map, excluded)
    manifest = Path(config["objaverse"]["nas_root"]) / "manifest.json"
    local = json.loads(manifest.read_text())["uid_to_categories"] if manifest.is_file() else {}
    pairs = cohorts.candidates(config["cohort"], lvis, seen, unseen, excluded, local)
    run.write("candidates.json", {"seen": seen, "unseen": unseen, "pairs": pairs,
                                  "excluded": len(excluded)})
    log(f"[select] {len(pairs)} candidates; seen={seen}; unseen={unseen}")


def download(run, log):
    if (run.dir / "download.json").is_file():
        return
    from viewactive.download import ensure_glbs, write_store

    objaverse = run.config["objaverse"]
    pairs = run.read("candidates.json")["pairs"]
    glbs, skipped = ensure_glbs([uid for uid, _ in pairs], cohorts.object_paths(objaverse["nas_root"]),
                                nas_root=objaverse["nas_root"], cache_root=objaverse["cache_root"],
                                max_mb=objaverse["max_mb"], workers=objaverse["download_workers"], log=log)
    write_store(run.store, glbs, dict(pairs))
    run.write("download.json", {"glbs": glbs, "skipped": skipped})
    log(f"[download] {len(glbs)} GLBs ready, {len(skipped)} skipped")


def canonicalize(run, log):
    if not run.canonical.is_file():
        objaverse = run.config["objaverse"]
        run_in_env("sam3d", "harness.canonicalize", ["--root", run.store, "--out", run.canonical,
                                                     "--max-mb", objaverse["max_mb"],
                                                     *(["--trust-file-up"] if objaverse["trust_file_up"] else [])],
                   gpu_id=run.config["gpu_id"], log=run.dir / "logs" / "canonicalize.log")


def cohort(run, log):
    if (run.dir / "objects.json").is_file():
        return
    candidates = run.read("candidates.json")
    canonical = json.loads(run.canonical.read_text())["objects"]
    objects = cohorts.finalize(run.config["cohort"], [tuple(p) for p in candidates["pairs"]], canonical,
                               candidates["seen"], candidates["unseen"], run.config["objaverse"]["max_faces"],
                               drop_suspect=not run.config["objaverse"]["trust_file_up"])
    run.write("objects.json", objects)
    counts = {}
    for record in objects.values():
        counts[record["split"]] = counts.get(record["split"], 0) + 1
    log(f"[cohort] {len(objects)} objects: {counts}")


def render(run, log):
    from harness.upright import verify_cameras

    settings = run.config["render"]
    if settings["backend"] == "blender":
        from viewactive.blender import render_shard
    for number, uids in enumerate(run.shards()):
        directory = run.shard_dir(number)
        report = run.dir / "logs" / f"camera-verification-{number:03d}.json"
        if report.is_file() and json.loads(report.read_text())["passed"]:
            continue
        if settings["backend"] == "blender":
            render_shard(run, number, uids, log)
        else:
            run_in_env("sam3d", "harness.build_dataset", [
                "--root", run.store, "--canonical", run.canonical, "--out", directory, "--uids", *uids,
                "--azimuths", *settings["azimuths"], "--elevations", *settings["elevations"],
                "--occlusion-levels", "0.0", "--distance", settings["distance"],
                "--image-size", settings["image_size"], "--max-faces", run.config["objaverse"]["max_faces"],
            ], gpu_id=run.config["gpu_id"], log=run.dir / "logs" / "render.log")
        expected = {(uid, float(az), float(el)) for uid in uids
                    for az in settings["azimuths"] for el in settings["elevations"]}
        verify_cameras(directory / "index.jsonl", report, expected)
        if not json.loads(report.read_text())["passed"]:
            raise RuntimeError(f"Camera verification failed for shard {number}: see {report}")
        log(f"[render] shard {number + 1}/{len(run.shards())} verified")


def label(run, log, workers):
    from viewactive.labels import label_object, normalize

    if run.vqf.is_file():
        return
    settings = run.config["labels"]
    objects, views = run.objects(), run.views()
    directory = run.dir / "labels"
    directory.mkdir(exist_ok=True)
    options = {**{k: settings[k] for k in ("samples", "seed", "polar_bins", "azimuth_bins", "visual_bins")},
               "azimuths": run.grid.azimuths, "elevations": run.grid.elevations,
               "distance": run.config["render"]["distance"], "nthreads": 4}
    tasks = [(uid, str(run.store), str(run.canonical), views[uid], options)
             for uid in objects if not (directory / f"{uid}.npy").is_file()]
    failures = {}
    if tasks:
        with multiprocessing.get_context("spawn").Pool(workers or settings["workers"]) as pool:
            for n, (uid, raw, error) in enumerate(pool.imap_unordered(label_object, tasks), 1):
                if error:
                    failures[uid] = error
                else:
                    np.save(directory / f"{uid}.npy", raw)
                if n % 10 == 0 or n == len(tasks):
                    log(f"[label] {n}/{len(tasks)} objects ({len(failures)} failed)")
    if failures:
        run.write("labels/failures.json", failures)
        raise RuntimeError(f"{len(failures)} objects failed labelling; see {directory / 'failures.json'}")
    uids = sorted(objects)
    raw = np.stack([np.load(directory / f"{uid}.npy") for uid in uids])
    np.savez(run.vqf, uids=np.array(uids), categories=np.array([objects[u]["category"] for u in uids]),
             splits=np.array([objects[u]["split"] for u in uids]), raw=raw,
             norm=normalize(raw, settings["floor"]), azimuths=np.array(run.grid.azimuths),
             elevations=np.array(run.grid.elevations), channels=np.array(["visible_ratio", "normal_entropy",
                                                                          "visual_entropy"]))
    log(f"[label] {len(uids)} objects; undefined views: {float(np.isnan(raw).any(1).mean()):.4f}")


def rgb_images(run, uids):
    views = run.views()
    return [[[cell[0] for cell in row] for row in views[uid]] for uid in uids]


def mask_images(run, uids):
    views = run.views()
    return [[[cell[1] for cell in row] for row in views[uid]] for uid in uids]


def features(run, log):
    from viewactive.features import Encoders, cache_features

    data = run.labelled()
    path = run.features_path()
    path.parent.mkdir(exist_ok=True)
    done = path.with_suffix(".done.npy")
    if done.is_file() and np.load(done).all():
        return
    encoders = Encoders(run.config["encoders"]["dino"], run.config["encoders"]["clip"])
    cache_features(path, done, rgb_images(run, data["uids"]), encoders, run.config["encoders"]["batch"], log)


def classify(run, log):
    from viewactive.classify import classify_views

    data = run.labelled()
    evaluated = np.isin(data["splits"], ["test", "test_unseen"])
    uids = data["uids"][evaluated]
    margin = run.config["classification"]["crop_margin"]
    images = rgb_images(run, uids)
    masks = mask_images(run, uids) if margin is not None else None
    directory = run.dir / "classifier"
    directory.mkdir(exist_ok=True)
    resplit = run.dir / "resplit.json"
    # The resplit may have been prepared on the host, whose absolute paths
    # differ from the container's. The linked render cache identifies it here.
    reuse = (run.dir / "renders").resolve().parent if resplit.is_file() else None
    for name, model in run.config["classifiers"].items():
        path = directory / f"{name}.npz"
        if path.is_file():
            with np.load(path) as stored:
                if (not np.array_equal(stored["uids"], uids) or str(stored["model"]) != model):
                    raise ValueError(f"{path} does not match this test cohort or classifier")
            continue
        top1 = np.empty((len(uids), *run.grid.shape), dtype=np.int16)
        cached = {}
        if reuse is not None:
            previous_config = json.loads((reuse / "config.json").read_text())
            if (previous_config["classifiers"].get(name) != model or
                    previous_config["classification"] != run.config["classification"]):
                raise ValueError(f"Cannot reuse {name}: source classification settings differ")
            with np.load(reuse / "classifier" / f"{name}.npz") as stored:
                cached = dict(zip(stored["uids"], stored["top1"]))
        missing = [i for i, uid in enumerate(uids) if uid not in cached]
        for i, uid in enumerate(uids):
            if uid in cached:
                top1[i] = cached[uid]
        if missing:
            fresh = classify_views(model, [images[i] for i in missing], log=log,
                                   masks=[masks[i] for i in missing] if masks else None,
                                   crop_margin=margin)
            top1[missing] = fresh
        np.savez(path, uids=uids, top1=top1, model=model)
        log(f"[classify] {name} done ({len(cached)} cached, {len(missing)} inferred)")


def train(run, log, device="cuda"):
    from viewactive.train import train as fit

    history = run.checkpoint.parent / "history.json"
    try:
        complete = (run.checkpoint.is_file() and
                    len(json.loads(history.read_text())) == run.config["train"]["epochs"])
    except (OSError, ValueError):
        complete = False
    if run.check("train", strict=False) and complete:
        return
    data = run.labelled()
    features = np.load(run.features_path(), mmap_mode="r")
    encoders = (run.config["encoders"]["dino"], run.config["encoders"]["clip"])
    fit(run.config, features, data["norm"], data["splits"], run.checkpoint,
        encoders=encoders, grid=run.grid, device=device, log=log)
    run.check("train", record=True)


def evaluate(run, log, fps=True, device="cuda"):
    import torch
    from PIL import Image
    from viewactive.classify import correct as is_correct
    from viewactive.evaluate import evaluate as run_evaluation, measure_fps
    from viewactive.model import ViewActiveNet

    data = run.labelled()
    data["features"] = np.load(run.features_path(), mmap_mode="r")
    state = torch.load(run.checkpoint, map_location=device, weights_only=False)
    net = ViewActiveNet(state["dim"], state["hidden"], run.grid.shape).to(device)
    net.load_state_dict(state["state_dict"])
    net.eval()
    class_map = cohorts.load_class_map(run.config["cohort"]["class_map"])
    index = {uid: i for i, uid in enumerate(data["uids"])}
    classifiers = {}
    for name in run.config["classifiers"]:
        path = run.dir / "classifier" / f"{name}.npz"
        if not path.is_file():
            continue
        with np.load(path) as stored:
            correct = np.zeros((len(index), *run.grid.shape), dtype=bool)
            for uid, top1 in zip(stored["uids"], stored["top1"]):
                category = data["categories"][index[uid]]
                correct[index[uid]] = is_correct(top1, category, class_map)
            classifiers[name] = correct
    images = rgb_images(run, data["uids"])
    report, _ = run_evaluation(run.config, net, run.grid, data, run.dir / "eval", classifiers=classifiers,
                               images=images, device=device, log=log)
    if fps:
        with Image.open(images[0][0][0]) as image:
            report["fps_batch1"] = measure_fps(run.checkpoint, image.convert("RGB"), device=device)
        (run.dir / "eval" / "fields.json").write_text(json.dumps(report, indent=1) + "\n")
    log(f"[evaluate] {json.dumps(report)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--workers", type=int, help="processes for the label stage (default: labels.workers)")
    parser.add_argument("--no-fps", action="store_true", help="skip the end-to-end speed measurement")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="device for training and evaluation (CPU evaluation needs --no-fps)")
    parser.add_argument("--gpu", type=int, help="host GPU to use instead of the config's gpu_id")
    args = parser.parse_args()
    if args.device == "cpu" and "evaluate" in args.stages and not args.no_fps:
        parser.error("CPU evaluation requires --no-fps")
    no_huge_pages()
    config = settings_io.load_viewactive(args.config)
    if args.gpu is not None:
        config["gpu_id"] = args.gpu
    use_gpu(config["gpu_id"])
    run = Run(config)
    (run.dir / "logs").mkdir(parents=True, exist_ok=True)
    (run.dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    with open(run.dir / "logs" / "viewactive.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another ViewActive driver is running on {run.dir}") from None
        for stage in STAGES:
            if stage not in args.stages:
                continue
            stream = open(run.dir / "logs" / f"{stage}.log", "a", buffering=1)

            def log(message, stream=stream):
                print(message, flush=True)
                stream.write(message + "\n")

            log(f"[viewactive] stage {stage}")
            if stage in FINGERPRINT:
                run.check(FINGERPRINT[stage])
            steps = {"select": select, "download": download, "canonicalize": canonicalize,
                     "cohort": cohort, "render": render, "features": features, "classify": classify,
                     "train": lambda r, l: train(r, l, device=args.device),
                     "label": lambda r, l: label(r, l, args.workers),
                     "evaluate": lambda r, l: evaluate(r, l, fps=not args.no_fps, device=args.device)}
            try:
                steps[stage](run, log)
            finally:
                stream.close()
    print(f"[viewactive] done -> {run.dir}", flush=True)


if __name__ == "__main__":
    main()
