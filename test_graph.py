import pytest
from unittest.mock import patch, MagicMock
from graph import apply_context_patch, apply_context_patches, find_best_match, PatchingError
from graph import patch_model, read_model, time_table_optimiser
from model import TimetableInput

def test_find_best_match():
    lines = ["apple", "banana", "cherry", "date"]
    context = ["banana", "cherry"]
    assert find_best_match(lines, context) == 1

def test_apply_context_patch():
    original = "line1\nline2\nline3\nline4\n"
    patch = "<<<\nline2\nline3\n---\nnew line2\nnew line3\n>>>"
    expected = "line1\nnew line2\nnew line3\nline4\n"
    assert apply_context_patch(original, patch) == expected

def test_apply_context_patches():
    original = "line1\nline2\nline3\nline4\n"
    patches = "<<<\nline2\n---\nnew line2\n>>>\n<<<\nline4\n---\nnew line4\n>>>"
    expected = "line1\nnew line2\nline3\nnew line4\n"
    assert apply_context_patches(original, patches) == expected

def test_patch_model():
    with patch('builtins.open', MagicMock(read_data="original code")):
        with patch('graph.apply_context_patches', return_value="new code"):
            with patch('ast.parse'):
                with patch('importlib.util.spec_from_file_location'):
                    with patch('importlib.util.module_from_spec'):
                        result = patch_model("<<<\noriginal\n---\nnew\n>>>")
                        assert result == "Model successfully patched and reloaded."

def test_read_model():
    with patch('builtins.open', MagicMock(read_data="model code")):
        assert read_model() == "model code"

@patch('model.create_and_solve_timetable_model')
def test_time_table_optimiser(mock_create_and_solve):
    mock_create_and_solve.return_value = "Optimized timetable"
    input_data = {
        "Classes": ["C1", "C2"],
        "TimeSlots": ["T1", "T2"],
        "Teachers": ["Teacher1"],
        "Classrooms": ["Room1"]
    }
    result = time_table_optimiser(input_data)
    assert result == "Optimized timetable"
