"""Checks for the durable viewpoint experiment store."""
import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.viewpoint_sensitivity import store


class ViewpointStoreTests(unittest.TestCase):
    def row(self, azimuth=0):
        return {
            "schema_version": 2, "uid": "object-a", "category": "chair",
            "azimuth": azimuth, "elevation": 0, "vocabulary": ["chair", "table"],
            "prompt_results": {
                "chair": {"best_score": 0.8},
                "table": {"best_score": 0.2},
            },
            "top1_correct": True, "prediction": "chair",
        }

    def test_trial_is_durable_indexed_and_unique(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / store.DATABASE_NAME
            with store.connect(path) as db:
                store.ensure_vocabulary(db, ["chair", "table"])
                settings = {"categories": ["chair", "table"], "detection": {"score_threshold": 0.5}}
                store.ensure_query_config(db, settings)
                store.insert_trial(db, self.row())
                with self.assertRaises(ValueError):
                    store.insert_trial(db, self.row())
            with store.connect(path) as db:
                self.assertEqual(store.completed_keys(db), {("object-a", 0.0, 0.0)})
                self.assertEqual(store.load_trials(db), [self.row()])
                indexed = db.execute("SELECT score_minus_true, top1_correct FROM trials").fetchone()
                self.assertAlmostEqual(indexed[0], -0.6)
                self.assertEqual(indexed[1], 1)
                with self.assertRaises(ValueError):
                    store.ensure_vocabulary(db, ["table", "chair"])
                with self.assertRaisesRegex(ValueError, "Recognition settings changed"):
                    store.ensure_query_config(db, {**settings, "detection": {"score_threshold": 0.7}})

    def test_migration_and_export_preserve_legacy_content(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            source = output / "results.jsonl"
            source.write_text(json.dumps(self.row()) + "\n")
            aggregate = b'{"summary":{"chair":{}}}\n'
            npz = b"npz-example-bytes"
            (output / "aggregate.json").write_bytes(aggregate)
            (output / "aggregate.npz").write_bytes(npz)
            self.assertEqual(store.migrate(output), 1)
            self.assertFalse(source.exists())
            self.assertFalse((output / "aggregate.json").exists())
            with store.connect(store.database_path(output)) as db:
                self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                self.assertEqual(store.trial_count(db), 1)
            target = output / "exported"
            store.export(output, target)
            self.assertEqual(json.loads((target / "results.jsonl").read_text()), self.row())
            self.assertEqual((target / "aggregate.json").read_bytes(), aggregate)
            self.assertEqual((target / "aggregate.npz").read_bytes(), npz)


if __name__ == "__main__":
    unittest.main()
