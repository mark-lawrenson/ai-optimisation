import unittest
from graph import apply_context_patch, PatchingError

class TestApplyContextPatch(unittest.TestCase):

    def test_basic_patch(self):
        original = "line1\nline2\nline3\n"
        patch = "<<<\nline2\n---\nline2_modified\n>>>"
        expected = "line1\nline2_modified\nline3\n"
        self.assertEqual(apply_context_patch(original, patch), expected)

    def test_patch_with_indentation(self):
        original = "def func():\n    line1\n    line2\n    line3\n"
        patch = "<<<\n    line2\n---\n    line2_modified\n>>>"
        expected = "def func():\n    line1\n    line2_modified\n    line3\n"
        self.assertEqual(apply_context_patch(original, patch), expected)

    def test_patch_with_additional_indentation(self):
        original = "def func():\n    if True:\n        line1\n        line2\n"
        patch = "<<<\n        line1\n---\n        line1_modified\n>>>"
        expected = "def func():\n    if True:\n        line1_modified\n        line2\n"
        self.assertEqual(apply_context_patch(original, patch), expected)

    def test_patch_with_less_indentation(self):
        original = "def func():\n    line1\n    line2\n"
        patch = "<<<\n    line1\n---\nline1_modified\n>>>"
        expected = "def func():\nline1_modified\n    line2\n"
        self.assertEqual(apply_context_patch(original, patch), expected)

    def test_patch_not_found(self):
        original = "line1\nline2\nline3\n"
        patch = "<<<\nline4\n---\nline4_modified\n>>>"
        with self.assertRaises(PatchingError):
            apply_context_patch(original, patch)

    def test_malformed_patch(self):
        original = "line1\nline2\nline3\n"
        patch = "<<<\nline2\n---\nline2_modified"
        with self.assertRaises(PatchingError):
            apply_context_patch(original, patch)

if __name__ == "__main__":
    unittest.main()
