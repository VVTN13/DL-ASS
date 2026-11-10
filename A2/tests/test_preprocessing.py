"""Signal, label/alignment, and export checks for the CBraMod waveform pipeline."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import torch

from A2 import preprocessing as prep


class PreprocessingTests(unittest.TestCase):
    def test_full_keeps_every_epoch_and_ignores_sampling_seed(self):
        candidates = {"SC4002": [(float(i * 30), i % 5) for i in range(100)],
                      "SC4001": [(float(i * 30), i % 5) for i in range(120)]}
        full = prep.sample_participant_epochs(candidates, None)
        self.assertEqual(full, candidates)
        self.assertEqual(sum(map(len, full.values())), 220)
        self.assertEqual(full, prep.sample_participant_epochs(candidates, None, seed=1))
        self.assertEqual(sum(map(len, prep.sample_participant_epochs(candidates, 83).values())), 83)

    def test_stage_mapping_wake_margin_and_incomplete_epochs(self):
        annotations = mne.Annotations(
            [0, 1980, 2010, 2040, 2070, 2100, 2130, 2160, 2190],
            [1980, 30, 30, 30, 30, 30, 30, 30, 2100],
            ["Sleep stage W", "Sleep stage 1", "Sleep stage 2", "Sleep stage 3",
             "Sleep stage 4", "Sleep stage R", "Movement time", "Sleep stage ?", "Sleep stage W"],
        )
        epochs = prep.select_epochs(annotations, 4290)
        self.assertEqual(epochs[0], (180.0, 0))
        self.assertEqual(epochs[-1], (3900.0, 0))
        self.assertEqual(len(epochs), 123)
        self.assertEqual([label for _, label in epochs].count(3), 2)
        self.assertNotIn(2130, [onset for onset, _ in epochs])
        self.assertNotIn(2160, [onset for onset, _ in epochs])
        clipped = prep.select_epochs(annotations, 3905)
        self.assertEqual(len(clipped), 122)
        self.assertTrue(all(onset + 30 <= 3905 for onset, _ in clipped))

    def test_duplicate_scoring_is_rejected(self):
        annotations = mne.Annotations([0, 0], [30, 30], ["Sleep stage 1", "Sleep stage 2"])
        with self.assertRaisesRegex(ValueError, "overlapping"):
            prep.select_epochs(annotations, 60)

    def test_83_epochs_are_shared_across_nights_and_keep_rare_stages(self):
        candidates = {}
        for night, counts in [("SC4001", [90, 10, 50, 1, 20]),
                              ("SC4002", [10, 40, 30, 20, 10])]:
            stages = np.repeat(np.arange(5), counts)
            candidates[night] = [(float(i * 30), int(stage)) for i, stage in enumerate(stages)]
        candidates["SC4031"] = [(float(i * 30), i % 5) for i in range(12)]
        selected = prep.sample_participant_epochs(candidates)
        self.assertEqual(sum(len(selected[n]) for n in ["SC4001", "SC4002"]), 83)
        self.assertEqual(selected["SC4031"], candidates["SC4031"])
        self.assertEqual(selected, prep.sample_participant_epochs(dict(reversed(list(candidates.items())))))
        for night, epochs in selected.items():
            self.assertEqual(len(epochs), len(set(epochs)))
            self.assertEqual(epochs, sorted(epochs))
            self.assertTrue(set(epochs).issubset(candidates[night]))
            self.assertEqual({label for _, label in epochs}, {0, 1, 2, 3, 4})
        rare_epoch = next(epoch for epoch in candidates["SC4001"] if epoch[1] == 3)
        self.assertIn(rare_epoch, selected["SC4001"])

    def test_filter_scaling_resampling_and_channel_order(self):
        t = np.arange(9000) / 100
        fpz = 50e-6 * np.sin(2 * np.pi * 5 * t) + 30e-6 * np.sin(2 * np.pi * 45 * t)
        pz = 20e-6 * np.sin(2 * np.pi * 12 * t)
        raw = mne.io.RawArray(np.array([pz, fpz]), mne.create_info(prep.CHANNELS[::-1], 100, "eeg"), verbose="ERROR")
        waveform = prep.prepare_waveforms(raw)
        self.assertEqual(waveform.shape, (2, 18000))
        self.assertEqual(waveform.dtype, np.float32)
        # Measure interior sinusoid amplitudes independently of the preprocessing implementation.
        segment = waveform[:, 4000:8000]
        times = np.arange(segment.shape[1]) / 200

        def amplitude(channel, frequency):
            return 2 * abs(np.mean(segment[channel] * np.exp(-2j * np.pi * frequency * times)))

        self.assertAlmostEqual(amplitude(0, 5), 0.5, delta=0.01)  # 50 microvolts / 100
        self.assertAlmostEqual(amplitude(1, 12), 0.2, delta=0.01)
        self.assertLess(amplitude(0, 45), 0.005)
        self.assertLess(amplitude(1, 5), 0.005)

    def test_patch_boundaries_preserve_signal_order(self):
        waveform = np.arange(2 * 12000, dtype=np.float32).reshape(2, 12000)
        patches = prep.make_patches(waveform, [(0.0, 0), (30.0, 1)])
        self.assertEqual(tuple(patches.shape), (2, 2, 30, 200))
        np.testing.assert_array_equal(patches[0].numpy().reshape(2, 6000), waveform[:, :6000])
        np.testing.assert_array_equal(patches[1].numpy().reshape(2, 6000), waveform[:, 6000:])
        with self.assertRaisesRegex(ValueError, "outside"):
            prep.make_patches(waveform, [(31.0, 0)])

    def test_invalid_signal_is_rejected(self):
        raw = mne.io.RawArray(np.zeros((2, 6000)), mne.create_info(prep.CHANNELS, 100, "eeg"), verbose="ERROR")
        raw._data[0, 30] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite source"):
            prep.prepare_waveforms(raw)

    def test_reference_label_or_onset_mismatch_is_rejected(self):
        prep.check_reference("SC4001", [(0.0, 1)], {"SC4001": [(0.0, 1)]})
        with self.assertRaisesRegex(ValueError, "differ from"):
            prep.check_reference("SC4001", [(0.0, 2)], {"SC4001": [(0.0, 1)]})
        with self.assertRaisesRegex(ValueError, "differ from"):
            prep.check_reference("SC4001", [(30.0, 1)], {"SC4001": [(0.0, 1)]})

    def test_export_and_reload_keep_metadata_folds_and_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "edf"
            data.mkdir()
            header = bytearray(256)
            header[168:184] = b"24.04.8916.13.00"
            checksums = []
            for recording in ["SC4001", "SC4031"]:
                for suffix in ["E0-PSG.edf", "EC-Hypnogram.edf"]:
                    path = data / (recording + suffix)
                    path.write_bytes(header)
                    checksums.append(f"{prep.sha256(path)} sleep-cassette/{path.name}")
            manifest = root / "SHA256SUMS.txt"
            manifest.write_text("\n".join(checksums) + "\n")
            folds = root / "folds.csv"
            folds.write_text("participant,split,validation_fold\nSC400,development,1\nSC403,test,-1\n")
            annotations = mne.Annotations(
                [0, 30, 60, 90, 120], [30] * 5,
                ["Sleep stage W", "Sleep stage 1", "Sleep stage 2", "Sleep stage 3", "Sleep stage R"],
            )
            t = np.arange(15000) / 100
            signal = np.array([np.sin(2 * np.pi * 5 * t), np.sin(2 * np.pi * 10 * t)]) * 50e-6
            raw = mne.io.RawArray(signal, mne.create_info(prep.CHANNELS, 100, "eeg"), verbose="ERROR")
            reference = root / "reference.pt"
            torch.save({"class_map": prep.CLASS_MAP, "channels": prep.CHANNELS,
                        "groups": ["SC400"] * 5 + ["SC403"] * 5,
                        "recordings": ["SC4001"] * 5 + ["SC4031"] * 5,
                        "labels": torch.tensor(list(range(5)) * 2),
                        "onsets": torch.tensor([0., 30., 60., 90., 120.] * 2)}, reference)
            with patch.object(prep.mne.io, "read_raw_edf", side_effect=lambda *a, **k: raw.copy()), \
                    patch.object(prep.mne, "read_annotations", return_value=annotations):
                summary = prep.preprocess_shards(data, root / "out", folds_path=folds,
                                          reference_path=reference, checksums_path=manifest)
            self.assertEqual(summary["epochs"], 10)
            self.assertTrue(summary["cnn_reference_checked"])
            self.assertTrue(summary["archive_checksums_checked"])
            shard = torch.load(root / "out/SC4031.pt", weights_only=True, mmap=True)
            self.assertEqual(tuple(shard["patches"].shape), (5, 2, 30, 200))
            self.assertEqual(shard["patches"].dtype, torch.float32)
            self.assertEqual(shard["labels"].dtype, torch.int64)
            self.assertTrue(torch.isfinite(shard["patches"]).all())
            self.assertEqual((shard["split"], shard["validation_fold"]), ("test", -1))
            with (root / "out/index.csv").open() as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 10)
            self.assertEqual({r["participant"] for r in rows if r["split"] == "development"}, {"SC400"})
            prep.merge_shards(root / "out", root / "small.pt")
            combined = torch.load(root / "small.pt", weights_only=True, mmap=True)
            self.assertEqual(tuple(combined["patches"].shape), (10, 2, 30, 200))
            self.assertIsNone(combined["sampling"]["samples_per_participant"])
            self.assertEqual(combined["sampling"]["method"], "all eligible epochs")
            output_manifest = json.loads((root / "small.json").read_text())
            self.assertEqual(output_manifest["output_sha256"], prep.sha256(root / "small.pt"))
            self.assertEqual(combined["splits"], ["development"] * 5 + ["test"] * 5)
            self.assertEqual(combined["validation_folds"].tolist(), [1] * 5 + [-1] * 5)
            with (root / "small.csv").open() as source:
                manifest_rows = list(csv.DictReader(source))
            self.assertEqual(len(manifest_rows), 10)
            self.assertEqual([int(row["label"]) for row in manifest_rows], combined["labels"].tolist())
            with self.assertRaises(FileExistsError):
                prep.preprocess_shards(data, root / "out", folds_path=folds)
            # Invalid EDF clocks must fail before any reader or signal processing runs.
            hyp = data / "SC4001EC-Hypnogram.edf"
            altered = bytearray(header)
            altered[176:184] = b"17.13.00"
            hyp.write_bytes(altered)
            with self.assertRaisesRegex(ValueError, "start date/time differ"):
                prep.preprocess_shards(data, root / "bad_clock", recordings=["SC4001"], folds_path=folds)
            with self.assertRaisesRegex(ValueError, "checksum"):
                prep.preprocess_shards(data, root / "bad_hash", recordings=["SC4001"],
                                folds_path=folds, checksums_path=manifest)


if __name__ == "__main__":
    unittest.main()
