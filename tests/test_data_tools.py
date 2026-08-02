import unittest

from scripts.prepare_annotations import keyframe_only


class DataToolsTest(unittest.TestCase):
    def test_keyframe_filter(self) -> None:
        coco = {
            "images": [
                {"id": 1, "file_name": "a.jpg", "is_keyframe": True},
                {"id": 2, "file_name": "b.jpg", "is_keyframe": False},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1]},
                {"id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 1, 1]},
            ],
            "categories": [{"id": 1, "name": "bee"}],
        }
        filtered = keyframe_only(coco)
        self.assertEqual([image["id"] for image in filtered["images"]], [1])
        self.assertEqual(
            [annotation["id"] for annotation in filtered["annotations"]], [1]
        )


if __name__ == "__main__":
    unittest.main()
