import json
import tempfile
import unittest
from pathlib import Path

from scripts.infer import ROOT, convert_predictions, make_chunks, manifest_path


class InferenceToolsTest(unittest.TestCase):
    def test_manifest_paths_do_not_expose_machine_prefixes(self) -> None:
        self.assertEqual(
            manifest_path(ROOT / "configs/buzzspot/example.py"),
            "configs/buzzspot/example.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            external = Path(directory) / "checkpoint.pth"
            self.assertEqual(manifest_path(external), "<external>/checkpoint.pth")

    def test_chunks_cover_each_image_once(self) -> None:
        reference = {
            "images": [{"id": index} for index in range(1, 6)],
            "annotations": [
                {"id": index, "image_id": index, "category_id": 1}
                for index in range(1, 6)
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = make_chunks(reference, Path(directory), 2)
            chunks = [json.loads(path.read_text()) for path in paths]
        image_ids = [image["id"] for chunk in chunks for image in chunk["images"]]
        annotation_ids = [
            annotation["image_id"]
            for chunk in chunks
            for annotation in chunk["annotations"]
        ]
        self.assertEqual(sorted(image_ids), [1, 2, 3, 4, 5])
        self.assertEqual(sorted(annotation_ids), [1, 2, 3, 4, 5])

    def test_one_based_categories_are_validated(self) -> None:
        predictions = [
            {"image_id": 7, "category_id": 1, "bbox": [1, 2, 3, 4], "score": 0.9},
            {"image_id": 7, "category_id": 4, "bbox": [5, 6, 7, 8], "score": 0.8},
            {"image_id": 7, "category_id": 0, "bbox": [1, 1, 1, 1], "score": 0.1},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.json"
            path.write_text(json.dumps(predictions))
            converted = convert_predictions(path, {7})
        self.assertEqual([item["category_id"] for item in converted], [1, 4])


if __name__ == "__main__":
    unittest.main()
