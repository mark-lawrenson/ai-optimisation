import pytest
import sys
import os

# Ensure the directory containing graph.py is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph import apply_context_patch, PatchingError

def test_basic_patch():
    original = "line1\nline2\nline3\n"
    patch = "<<<\nline2\n---\nline2_modified\n>>>"
    expected = "line1\nline2_modified\nline3\n"
    assert apply_context_patch(original, patch) == expected

def test_patch_with_indentation():
    original = "def func():\n    line1\n    line2\n    line3\n"
    patch = "<<<\n    line2\n---\n    line2_modified\n>>>"
    expected = "def func():\n    line1\n    line2_modified\n    line3\n"
    assert apply_context_patch(original, patch) == expected

def test_patch_with_additional_indentation():
    original = "def func():\n    if True:\n        line1\n        line2\n"
    patch = "<<<\n        line1\n---\n        line1_modified\n>>>"
    expected = "def func():\n    if True:\n        line1_modified\n        line2\n"
    assert apply_context_patch(original, patch) == expected

def test_patch_with_less_indentation():
    original = "def func():\n    line1\n    line2\n"
    patch = "<<<\n    line1\n---\nline1_modified\n>>>"
    expected = "def func():\nline1_modified\n    line2\n"
    assert apply_context_patch(original, patch) == expected

def test_patch_not_found():
    original = "line1\nline2\nline3\n"
    patch = "<<<\nline4\n---\nline4_modified\n>>>"
    with pytest.raises(PatchingError):
        apply_context_patch(original, patch)

def test_malformed_patch():
    original = "line1\nline2\nline3\n"
    patch = "<<<\nline2\n---\nline2_modified"
    with pytest.raises(PatchingError):
        apply_context_patch(original, patch)
