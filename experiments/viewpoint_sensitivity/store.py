"""Durable SQLite store for viewpoint trials and derived analysis.

One transaction per recognition trial allows query workers to resume after a
crash. Detailed trial payloads are zlib-compressed; indexed scalar columns
support filtering without decompressing every row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import zlib
from pathlib import Path

DATABASE_NAME = "experiment.sqlite"
SCHEMA_VERSION = 1
TRIAL_SCHEMA_VERSION = 2


def database_path(output):
    return Path(output) / DATABASE_NAME


def connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA synchronous=FULL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY, value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trials (
            uid TEXT NOT NULL,
            azimuth REAL NOT NULL,
            elevation REAL NOT NULL,
            category TEXT NOT NULL,
            top1_correct INTEGER,
            score_minus_true REAL,
            payload BLOB NOT NULL,
            PRIMARY KEY (uid, azimuth, elevation)
        );
        CREATE INDEX IF NOT EXISTS trials_by_category_view
            ON trials(category, elevation, azimuth);
        CREATE TABLE IF NOT EXISTS artifacts (
            name TEXT PRIMARY KEY, payload BLOB NOT NULL
        );
    """)
    version = db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if version is None:
        db.execute("INSERT INTO meta VALUES (?, ?)", ("schema_version", str(SCHEMA_VERSION)))
        db.commit()
    elif int(version[0]) != SCHEMA_VERSION:
        db.close()
        raise ValueError(f"Unsupported experiment database schema {version[0]}: {path}")
    return db


def _pack_json(value):
    return zlib.compress(json.dumps(value, separators=(",", ":"), allow_nan=False).encode(), 6)


def _unpack_json(payload):
    return json.loads(zlib.decompress(payload))


def vocabulary(db):
    row = db.execute("SELECT value FROM meta WHERE key='vocabulary'").fetchone()
    return json.loads(row[0]) if row else None


def ensure_vocabulary(db, categories):
    categories = list(categories)
    existing = vocabulary(db)
    if existing is not None and existing != categories:
        raise ValueError("Experiment vocabulary changed; restart the query in a new database")
    if existing is None:
        with db:
            db.execute("INSERT INTO meta VALUES (?, ?)", ("vocabulary", json.dumps(categories)))


def ensure_query_config(db, config):
    """Reject resuming trials with a changed recognition setup."""
    relevant = {key: config.get(key) for key in
                ("categories", "detection", "checkpoint", "render_dir")}
    fingerprint = hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()
    existing = db.execute("SELECT value FROM meta WHERE key='query_fingerprint'").fetchone()
    if existing and existing[0] != fingerprint:
        raise ValueError("Recognition settings changed; restart the query before resuming")
    if not existing:
        with db:
            db.execute("INSERT INTO meta VALUES (?, ?)", ("query_fingerprint", fingerprint))


def _trial_fields(row):
    if row.get("schema_version") != TRIAL_SCHEMA_VERSION:
        raise ValueError("Recognition trial uses an unsupported schema")
    scores = row["prompt_results"]
    category = row["category"]
    true_score = float(scores[category]["best_score"])
    wrong_score = max(float(result["best_score"]) for prompt, result in scores.items()
                      if prompt != category)
    correct = row.get("top1_correct")
    return (row["uid"], float(row["azimuth"]), float(row["elevation"]), category,
            None if correct is None else int(correct), wrong_score - true_score,
            _pack_json(row))


def insert_trial(db, row):
    current = vocabulary(db)
    if current is None or row.get("vocabulary") != current:
        raise ValueError("Recognition trial vocabulary does not match the experiment")
    try:
        with db:
            db.execute("INSERT INTO trials VALUES (?, ?, ?, ?, ?, ?, ?)", _trial_fields(row))
    except sqlite3.IntegrityError as error:
        raise ValueError("Duplicate recognition trial for this object and viewpoint") from error


def completed_keys(db):
    return set(db.execute("SELECT uid, azimuth, elevation FROM trials"))


def load_trials(db):
    return [_unpack_json(payload) for (payload,) in db.execute(
        "SELECT payload FROM trials ORDER BY category, uid, elevation, azimuth")]


def trial_count(db):
    return db.execute("SELECT COUNT(*) FROM trials").fetchone()[0]


def clear_trials(db):
    with db:
        db.execute("DELETE FROM trials")
        db.execute("DELETE FROM artifacts")
        db.execute("DELETE FROM meta WHERE key IN ('vocabulary', 'query_fingerprint')")


def put_artifact(db, name, payload):
    if isinstance(payload, (dict, list)):
        payload = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    with db:
        db.execute("INSERT OR REPLACE INTO artifacts VALUES (?, ?)",
                   (name, zlib.compress(payload, 6)))


def get_artifact(db, name):
    row = db.execute("SELECT payload FROM artifacts WHERE name=?", (name,)).fetchone()
    return zlib.decompress(row[0]) if row else None


def get_analysis(db):
    payload = get_artifact(db, "aggregate.json")
    return json.loads(payload) if payload else None


def migrate(output):
    """Import legacy files to a new database, validating before replacing them."""
    output = Path(output)
    legacy = output / "results.jsonl"
    if not legacy.is_file():
        raise FileNotFoundError(legacy)
    destination = database_path(output)
    if destination.exists():
        raise FileExistsError(f"Database already exists: {destination}")
    temp = output / f".{DATABASE_NAME}.tmp"
    if temp.exists():
        temp.unlink()
    db = connect(temp)
    count = 0
    try:
        expected_vocabulary = None
        with legacy.open() as stream:
            for number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    if expected_vocabulary is None:
                        expected_vocabulary = row["vocabulary"]
                        ensure_vocabulary(db, expected_vocabulary)
                    if row["vocabulary"] != expected_vocabulary:
                        raise ValueError("Vocabulary changes between trials")
                    db.execute("INSERT INTO trials VALUES (?, ?, ?, ?, ?, ?, ?)",
                               _trial_fields(row))
                except Exception as error:
                    raise ValueError(f"Invalid trial at {legacy}:{number}") from error
                count += 1
        db.commit()
        if count == 0 or count != trial_count(db):
            raise ValueError("Empty or incomplete trial import")
        settings = output / "settings.json"
        if settings.is_file():
            ensure_query_config(db, json.loads(settings.read_text()))
        for name in ("aggregate.json", "aggregate.npz"):
            source = output / name
            if source.is_file():
                put_artifact(db, name, source.read_bytes())
                if get_artifact(db, name) != source.read_bytes():
                    raise ValueError(f"Artifact mismatch after database import: {name}")
        check = db.execute("PRAGMA integrity_check").fetchone()[0]
        if check != "ok":
            raise ValueError(f"SQLite integrity check failed: {check}")
        db.close()
        temp.replace(destination)
        for name in ("results.jsonl", "aggregate.json", "aggregate.npz"):
            (output / name).unlink(missing_ok=True)
        return count
    except Exception:
        db.close()
        temp.unlink(missing_ok=True)
        raise


def export(output, target):
    """Recreate the previous machine-readable files on demand."""
    output, target = Path(output), Path(target)
    if not database_path(output).is_file():
        raise FileNotFoundError(database_path(output))
    target.mkdir(parents=True, exist_ok=True)
    with connect(database_path(output)) as db:
        with (target / "results.jsonl").open("w") as stream:
            for row in load_trials(db):
                stream.write(json.dumps(row) + "\n")
        for name in ("aggregate.json", "aggregate.npz"):
            payload = get_artifact(db, name)
            if payload is not None:
                (target / name).write_bytes(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("migrate", "export", "stats"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--target", type=Path,
                        help="Export directory, required for export")
    args = parser.parse_args()
    if args.action == "migrate":
        print(f"Imported {migrate(args.output)} trials into {database_path(args.output)}")
    elif args.action == "export":
        if args.target is None:
            parser.error("--target is required for export")
        export(args.output, args.target)
        print(args.target)
    else:
        with connect(database_path(args.output)) as db:
            print(json.dumps({"trials": trial_count(db),
                              "artifacts": [r[0] for r in db.execute("SELECT name FROM artifacts")]}))


if __name__ == "__main__":
    main()
