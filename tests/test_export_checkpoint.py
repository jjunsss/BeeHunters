import unittest

from scripts.export_inference_checkpoint import public_meta


class ExportCheckpointTest(unittest.TestCase):
    def test_public_meta_is_allowlisted(self) -> None:
        dataset_meta = {"classes": ("bee", "bumblebee", "hoverfly", "moth")}
        checkpoint = {
            "meta": {
                "cfg": "work_dir='/private/path'",
                "experiment_name": "internal-run",
                "time": "private-timestamp",
                "dataset_meta": dataset_meta,
            }
        }
        self.assertEqual(public_meta(checkpoint), {"dataset_meta": dataset_meta})

    def test_dataset_meta_is_required(self) -> None:
        with self.assertRaises(KeyError):
            public_meta({"meta": {"cfg": "config text"}})


if __name__ == "__main__":
    unittest.main()
