import unittest
from unittest.mock import patch, MagicMock
from pyomo.environ import ConcreteModel, Var
from model import create_and_solve_timetable_model, TimetableInput

class TestModel(unittest.TestCase):

    def setUp(self):
        self.data = {
            "Classes": ["C1", "C2"],
            "TimeSlots": ["T1", "T2"],
            "Teachers": ["Teacher1", "Teacher2"],
            "Classrooms": ["Room1", "Room2"]
        }

    @patch('model.Highs')
    def test_create_and_solve_timetable_model(self, mock_highs):
        mock_solver = MagicMock()
        mock_highs.return_value = mock_solver
        mock_results = MagicMock()
        mock_results.termination_condition = "optimal"
        mock_results.best_objective_bound = 2
        mock_results.best_feasible_objective = 2
        mock_solver.solve.return_value = mock_results

        with patch.object(ConcreteModel, 'create_instance', return_value=ConcreteModel()):
            with patch.object(Var, 'get_values', return_value={('C1', 'T1'): 1, ('C2', 'T2'): 1}):
                result = create_and_solve_timetable_model(self.data)

        self.assertIn('room_table', result)
        self.assertIn('teacher_table', result)
        self.assertIn('debug_info', result)
        self.assertEqual(result['debug_info']['termination_condition'], "optimal")

    def test_timetable_input_validation(self):
        valid_input = TimetableInput(**self.data)
        self.assertEqual(valid_input.Classes, ["C1", "C2"])
        self.assertEqual(valid_input.TimeSlots, ["T1", "T2"])
        self.assertEqual(valid_input.Teachers, ["Teacher1", "Teacher2"])
        self.assertEqual(valid_input.Classrooms, ["Room1", "Room2"])

        with self.assertRaises(ValueError):
            TimetableInput(Classes=["C1"], TimeSlots=["T1"], Teachers=[], Classrooms=["Room1"])

if __name__ == '__main__':
    unittest.main()
