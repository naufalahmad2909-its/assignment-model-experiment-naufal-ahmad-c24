"""Pengujian offline untuk split, provenance, prediksi, dan metrik."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import experiment as exp


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.df, self.info = exp.load_dataset()
        self.train, self.test, self.split = exp.split_dataset(self.df)

    def test_group_split_has_no_text_or_id_leakage_and_keeps_all_rows(self):
        self.assertFalse(set(self.train.text_group) & set(self.test.text_group))
        self.assertFalse(set(self.train.review_id) & set(self.test.review_id))
        self.assertEqual(set(self.df.review_id), set(self.train.review_id) | set(self.test.review_id))
        self.assertEqual((len(self.train), len(self.test)), (158, 42))
        self.assertEqual((self.train.text_group.nunique(), self.test.text_group.nunique()), (32, 8))

    def test_raw_integrity_and_split_reproducibility(self):
        report = json.loads((exp.DOCS / "experiment_report.json").read_text(encoding="utf-8"))
        self.assertEqual(self.info["raw_sha256"], report["inspection"]["raw_sha256"])
        second_train, second_test, _ = exp.split_dataset(self.df)
        self.assertEqual(self.train.review_id.tolist(), second_train.review_id.tolist())
        self.assertEqual(self.test.review_id.tolist(), second_test.review_id.tolist())

    def test_tfidf_is_fit_on_training_text_only(self):
        vectorizer, classifier, _ = exp.train_classic(self.train)
        train_tokens = set(token for text in self.train.review_text for token in vectorizer.build_analyzer()(text))
        self.assertEqual(set(vectorizer.get_feature_names_out()), train_tokens)
        pred, _ = exp.predict_classic(vectorizer, classifier, self.test)
        self.assertEqual(len(pred), len(self.test))
        self.assertTrue(set(pred).issubset(exp.LABELS))

    def test_response_normalization_rejects_unrecognized_output(self):
        self.assertEqual(exp.normalize_prediction(' "POSITIF"\n'), 'positif')
        with self.assertRaises(ValueError):
            exp.normalize_prediction('positif atau negatif')

    def test_recorded_api_covers_same_test_ids_and_rejects_changed_identity(self):
        labels, payload, mode = exp.predict_gemini(self.test)
        self.assertEqual(mode, "replay_recorded_real_api_responses")
        self.assertEqual(len(labels), len(self.test))
        self.assertEqual(len(payload["responses"]), 8)
        self.assertTrue(all(item["response_id"] for item in payload["responses"]))
        self.assertEqual(payload["identity"]["test_review_ids"], self.test.review_id.tolist())
        payload["identity"]["prompt_sha256"] = "wrong-prompt"
        with tempfile.TemporaryDirectory(dir=ROOT / "tests", prefix=".tmp-") as directory:
            cache = Path(directory) / "cache.json"
            cache.write_text(json.dumps(payload), encoding="utf-8")
            with patch.object(exp, "CACHE_PATH", cache):
                with self.assertRaisesRegex(ValueError, "Cache tidak cocok"):
                    exp.predict_gemini(self.test)

    def test_saved_metrics_match_independent_counts_and_no_test_rows_dropped(self):
        results = pd.read_csv(exp.DOCS / "test_predictions.csv")
        report = json.loads((exp.DOCS / "experiment_report.json").read_text(encoding="utf-8"))
        self.assertEqual(results.review_id.tolist(), self.test.review_id.tolist())
        self.assertEqual(results.sentiment.tolist(), self.test.sentiment.tolist())
        for name in ["classic", "gemini"]:
            counts = np.zeros((2, 2), dtype=int)
            for actual, pred in zip(results.sentiment, results[f"prediction_{name}"]):
                counts[exp.LABELS.index(actual), exp.LABELS.index(pred)] += 1
            metrics = report["metrics_on_all_test_rows"][name]
            self.assertEqual(counts.tolist(), metrics["confusion_matrix"])
            precision, recall, f1 = [], [], []
            for i in range(2):
                p = counts[i, i] / counts[:, i].sum() if counts[:, i].sum() else 0
                r = counts[i, i] / counts[i, :].sum() if counts[i, :].sum() else 0
                precision.append(p)
                recall.append(r)
                f1.append(2*p*r/(p+r) if p+r else 0)
            self.assertAlmostEqual(float(counts.trace()/counts.sum()), metrics["accuracy"])
            self.assertAlmostEqual(float(np.mean(precision)), metrics["precision_macro"])
            self.assertAlmostEqual(float(np.mean(recall)), metrics["recall_macro"])
            self.assertAlmostEqual(float(np.mean(f1)), metrics["f1_macro"])


if __name__ == "__main__":
    unittest.main()
