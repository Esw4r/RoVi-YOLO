"""Regression checks for the trained temporal pipeline; run with unittest."""
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from .model import DEFAULT_CHECKPOINT, DEFAULT_CNN, TemporalPredictor, sha256
from .train import windows


class TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.predictor = TemporalPredictor(DEFAULT_CHECKPOINT, sha256(DEFAULT_CNN))
        cls.manifest = json.loads((DEFAULT_CHECKPOINT.parent/'manifest.json').read_text())
        cls.data = windows(cls.manifest['records'], 'val', cls.predictor.config['sequence_length'])

    def test_online_scores_match_offline_sequence(self):
        sequence = self.data[0][0]
        window = self.predictor.new_window()
        for i, feature in enumerate(sequence):
            result = window.push(feature, i / 5)
            self.assertEqual(result['ready'], i == len(sequence)-1)
        with torch.inference_mode():
            expected = self.predictor.model(sequence[None]).sigmoid()[0]
        for row in result['scores']:
            if row['supported']:
                self.assertAlmostEqual(row['score'], expected[row['class_id']].item(), places=6)
            else:
                self.assertIsNone(row['score'])
                self.assertFalse(row['active'])

    def test_restart_gap_and_connections_clear_history(self):
        a, b = self.predictor.new_window(), self.predictor.new_window()
        for i, feature in enumerate(self.data[0][0]):
            a.push(feature, i / 5)
        self.assertEqual(len(b.frames), 0)
        self.assertFalse(a.push(self.data[0][0][0], 0)['ready'])
        self.assertEqual(len(a.frames), 1)
        a.push(self.data[0][0][1], 10)
        self.assertEqual(len(a.frames), 1)
        a.reset()
        self.assertEqual(len(a.frames), 0)

    def test_fast_duplicate_frames_not_added(self):
        w = self.predictor.new_window()
        f = self.data[0][0][0]
        w.push(f, 0)
        w.push(f, 0.05)
        self.assertEqual(len(w.frames), 1)
        with self.assertRaises(ValueError):
            w.push(f, float('nan'))

    def test_wrong_cnn_rejected(self):
        with self.assertRaisesRegex(ValueError, 'CNN'):
            TemporalPredictor(DEFAULT_CHECKPOINT, cnn_hash='wrong-checkpoint')

    def test_split_boundaries_and_normalization(self):
        rows = self.manifest['records']
        self.assertEqual(len({r['fingerprint']['video_sha256'] for r in rows}), 8)
        for split in ('train', 'val', 'test', 'night_test'):
            x, y = windows(rows, split, 8)
            expected_count = sum(r['sampled_frames'] - 7 for r in rows if r['split'] == split)
            self.assertEqual(len(x), expected_count)
            self.assertEqual(len(y), expected_count)
        raw = np.concatenate([np.load(r['cache'])['features'] for r in rows if r['split'] == 'train'])
        np.testing.assert_allclose(self.predictor.model.mean.numpy(), raw.mean(0), atol=1e-6)

    def test_model_uses_history(self):
        # Keep the final frame identical: earlier frames must influence the model.
        a = self.data[0][:1].clone()
        b = a.clone()
        b[:, :-1] = 0
        with torch.inference_mode():
            self.assertFalse(torch.allclose(self.predictor.model(a), self.predictor.model(b)))


if __name__ == '__main__':
    unittest.main()
