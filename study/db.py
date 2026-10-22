"""One SQLite database per study: images, SAM3 detections, and ambiguity scores.

``images`` holds both ground-truth renders (``source='gt'``) and SAM3D candidate
renders (``source='sam3d'``, ``parent_id`` = the ground-truth input image).
Azimuth/elevation are in the canonical object frame for ``gt`` and in the
reconstruction frame for ``sam3d`` (see tools/cameras.py). Paths are relative
to the study output directory.

``detections`` has one row per image and vocabulary prompt: the maximum SAM3
candidate confidence (0 when there is no candidate). The true-category row also
stores every candidate's score and mask IoU, so thresholds can be re-swept.

Every write commits per image, so interrupted stages resume where they stopped.
"""
import json
import sqlite3
import zlib
from pathlib import Path

NAME = "study.sqlite"
SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY,
    key TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL CHECK (source IN ('gt', 'sam3d')),
    parent_id INTEGER REFERENCES images(image_id),
    uid TEXT NOT NULL,
    category TEXT NOT NULL,
    azimuth REAL NOT NULL,
    elevation REAL NOT NULL,
    path TEXT NOT NULL,
    mask_path TEXT,
    visible_px INTEGER NOT NULL,
    meta BLOB
);
CREATE INDEX IF NOT EXISTS images_by_parent ON images(parent_id);
CREATE INDEX IF NOT EXISTS images_by_object ON images(source, category, uid);
CREATE TABLE IF NOT EXISTS detections (
    image_id INTEGER NOT NULL REFERENCES images(image_id),
    prompt TEXT NOT NULL,
    score REAL NOT NULL,
    n_candidates INTEGER NOT NULL,
    candidates BLOB,
    PRIMARY KEY (image_id, prompt)
);
CREATE TABLE IF NOT EXISTS view_scores (
    image_id INTEGER NOT NULL REFERENCES images(image_id),
    metric TEXT NOT NULL,
    value REAL,
    PRIMARY KEY (image_id, metric)
);
CREATE TABLE IF NOT EXISTS reconstructions (
    input_id INTEGER PRIMARY KEY REFERENCES images(image_id),
    status TEXT NOT NULL CHECK (status IN ('ok', 'error')),
    directory TEXT,
    seconds REAL,
    error TEXT
);
-- Written by the pick stage, before detection: the candidate each pick
-- selector chooses for each input; meta holds its angles and target.
CREATE TABLE IF NOT EXISTS picks (
    input_id INTEGER NOT NULL REFERENCES images(image_id),
    selector TEXT NOT NULL,
    pick_id INTEGER NOT NULL REFERENCES images(image_id),
    meta BLOB,
    PRIMARY KEY (input_id, selector)
);
-- Derived by the analyze stage (rebuilt on every run): the outcome of each
-- selector for each input. 'input' is the unmodified ground-truth view (before).
-- Values are weighted means over the selector's picks (random = expectation).
CREATE TABLE IF NOT EXISTS selections (
    input_id INTEGER NOT NULL REFERENCES images(image_id),
    selector TEXT NOT NULL,
    pick_id INTEGER REFERENCES images(image_id),
    margin REAL NOT NULL,
    correct REAL NOT NULL,
    fp_rate REAL NOT NULL,
    any_fp REAL NOT NULL,
    tp REAL NOT NULL,
    PRIMARY KEY (input_id, selector)
);
"""


def pack(value):
    return zlib.compress(json.dumps(value, separators=(",", ":"), allow_nan=False).encode(), 6)


def unpack(payload):
    return None if payload is None else json.loads(zlib.decompress(payload))


def connect(output_dir):
    path = Path(output_dir) / NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=60)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    version = get_meta(db, "schema_version")
    if version is None:
        set_meta(db, "schema_version", SCHEMA_VERSION)
    elif version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported study database schema {version}: {path}")
    return db


def get_meta(db, key):
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return None if row is None else json.loads(row[0])


def set_meta(db, key, value):
    with db:
        db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))


def ensure_meta(db, key, value, hint):
    """Store ``value`` once; later runs must match it."""
    existing = get_meta(db, key)
    if existing is None:
        set_meta(db, key, value)
    elif existing != value:
        raise ValueError(f"{key} changed since this study was created ({existing} -> {value}). {hint}")


def add_image(db, *, key, source, uid, category, azimuth, elevation, path, visible_px,
              mask_path=None, parent_id=None, meta=None):
    """Insert an image if its key is new; return its id. The caller commits."""
    db.execute(
        "INSERT OR IGNORE INTO images (key, source, parent_id, uid, category, azimuth, elevation,"
        " path, mask_path, visible_px, meta) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (key, source, parent_id, uid, category, float(azimuth), float(elevation), str(path),
         None if mask_path is None else str(mask_path), int(visible_px),
         None if meta is None else pack(meta)))
    return db.execute("SELECT image_id FROM images WHERE key=?", (key,)).fetchone()[0]


def images(db, where="1", params=()):
    """Image rows as dicts, ordered by object then view.

    Every query helper aliases images as ``i``, so one ``where`` clause (for
    example ``"i.source='gt'"``) selects the same images in all of them.
    """
    return [dict(row) for row in db.execute(
        "SELECT i.image_id, i.key, i.source, i.parent_id, i.uid, i.category, i.azimuth,"
        f" i.elevation, i.path, i.mask_path, i.visible_px FROM images i WHERE {where}"
        " ORDER BY i.category, i.uid, i.parent_id, i.elevation, i.azimuth", params)]


def add_detections(db, image_id, detections):
    """``detections`` maps prompt -> (score, n_candidates, candidates or None)."""
    with db:
        db.executemany("INSERT OR REPLACE INTO detections VALUES (?, ?, ?, ?, ?)",
                       [(image_id, prompt, float(score), int(count),
                         None if candidates is None else pack(candidates))
                        for prompt, (score, count, candidates) in detections.items()])


def add_scores(db, scores):
    """``scores`` is a list of (image_id, metric, value or None)."""
    with db:
        db.executemany("INSERT OR REPLACE INTO view_scores VALUES (?, ?, ?)", scores)


def prompt_scores(db, where="1", params=()):
    """{image_id: {prompt: max score}} for the selected images."""
    scores = {}
    for image_id, prompt, score in db.execute(
            "SELECT d.image_id, d.prompt, d.score FROM detections d JOIN images i USING (image_id)"
            f" WHERE {where}", params):
        scores.setdefault(image_id, {})[prompt] = score
    return scores


def true_candidates(db, where="1", params=()):
    """{image_id: {"scores": [...], "ious": [...]}} for each image's true-category prompt."""
    return {image_id: unpack(payload) for image_id, payload in db.execute(
        "SELECT d.image_id, d.candidates FROM detections d JOIN images i USING (image_id)"
        f" WHERE d.prompt = i.category AND {where}", params)}


def view_scores(db, where="1", params=()):
    """{image_id: {metric: value}} for the selected images."""
    scores = {}
    for image_id, metric, value in db.execute(
            "SELECT s.image_id, s.metric, s.value FROM view_scores s JOIN images i USING (image_id)"
            f" WHERE {where}", params):
        scores.setdefault(image_id, {})[metric] = value
    return scores
