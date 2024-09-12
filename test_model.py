import pytest
from unittest.mock import patch, MagicMock
from pyomo.environ import ConcreteModel, Var, TerminationCondition
from model import create_and_solve_timetable_model, TimetableInput
from pydantic import ValidationError

@pytest.fixture
def sample_data():
    return {
        "Classes": ["C1", "C2"],
        "TimeSlots": ["T1", "T2"],
        "Teachers": ["Teacher1", "Teacher2"],
        "Classrooms": ["Room1", "Room2"]
    }

@patch('model.Highs')
def test_create_and_solve_timetable_model(mock_highs, sample_data):
    mock_solver = MagicMock()
    mock_highs.return_value = mock_solver
    mock_results = MagicMock()
    mock_results.termination_condition = TerminationCondition.optimal
    mock_results.best_objective_bound = 2
    mock_results.best_feasible_objective = 2
    mock_solver.solve.return_value = mock_results

    with patch.object(ConcreteModel, 'create_instance', return_value=ConcreteModel()):
        with patch.object(Var, 'get_values', return_value={('C1', 'T1'): 1, ('C2', 'T2'): 1}):
            result = create_and_solve_timetable_model(sample_data)

    assert 'room_table' in result
    assert 'teacher_table' in result
    assert 'debug_info' in result
    assert result['debug_info']['termination_condition'] == TerminationCondition.optimal

def test_timetable_input_validation(sample_data):
    valid_input = TimetableInput(**sample_data)
    assert valid_input.Classes == ["C1", "C2"]
    assert valid_input.TimeSlots == ["T1", "T2"]
    assert valid_input.Teachers == ["Teacher1", "Teacher2"]
    assert valid_input.Classrooms == ["Room1", "Room2"]

    with pytest.raises(ValidationError):
        TimetableInput(Classes=["C1"], TimeSlots=["T1"], Teachers=[], Classrooms=["Room1"])
