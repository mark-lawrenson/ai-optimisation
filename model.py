from prettytable import PrettyTable
from pyomo.environ import *
from pydantic import BaseModel
from typing import List
from pyomo.contrib.appsi.solvers import Highs
import time
from loguru import logger
import sys

# Configure loguru
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("debug_model.log", rotation="500 MB", level="DEBUG")

class TimetableInput(BaseModel):
    Classes: List[str]
    TimeSlots: List[str]
    Teachers: List[str]
    Classrooms: List[str]


class TimetableInputSchema(BaseModel):
    input: TimetableInput


# Create a model
model = ConcreteModel()


def create_and_solve_timetable_model(data):
    # Create a model
    model = ConcreteModel()

    # Sets
    model.Classes = Set(initialize=data["Classes"])
    model.TimeSlots = Set(initialize=list(range(1, len(data["TimeSlots"]) + 1)))
    model.Teachers = Set(initialize=data["Teachers"])
    model.Classrooms = Set(initialize=data["Classrooms"])

    # Decision Variables
    model.x = Var(model.Classes, model.TimeSlots, domain=Binary)
    model.teacher_assignment = Var(model.Classes, model.Teachers, domain=Binary)
    model.room_assignment = Var(model.Classes, model.Classrooms, domain=Binary)
    model.z = Var(model.Classes, model.TimeSlots, model.Teachers, domain=Binary)
    model.w = Var(model.Classes, model.TimeSlots, model.Classrooms, domain=Binary)

    # Constraints
    # 1. Each class must be assigned exactly one time slot
    def one_time_slot_per_class_rule(model, c):
        return sum(model.x[c, t] for t in model.TimeSlots) == 1

    model.one_time_slot_per_class = Constraint(
        model.Classes, rule=one_time_slot_per_class_rule
    )

    # 2. Each class must be assigned exactly one teacher
    def one_teacher_per_class_rule(model, c):
        return (
            sum(model.teacher_assignment[c, teacher] for teacher in model.Teachers) == 1
        )

    model.one_teacher_per_class = Constraint(
        model.Classes, rule=one_teacher_per_class_rule
    )

    # 3. Each class must be assigned exactly one room
    def one_room_per_class_rule(model, c):
        return sum(model.room_assignment[c, room] for room in model.Classrooms) == 1

    model.one_room_per_class = Constraint(model.Classes, rule=one_room_per_class_rule)

    # 4. No two classes that share a common teacher can be in the same time slot
    def teacher_conflict_rule(model, t, teacher):
        return sum(model.z[c, t, teacher] for c in model.Classes) <= 1

    model.teacher_conflict = Constraint(
        model.TimeSlots, model.Teachers, rule=teacher_conflict_rule
    )

    # Link z with x and teacher_assignment
    def link_z_with_x_and_teacher_rule1(model, c, t, teacher):
        return model.z[c, t, teacher] <= model.x[c, t]

    model.link_z_with_x_and_teacher1 = Constraint(
        model.Classes,
        model.TimeSlots,
        model.Teachers,
        rule=link_z_with_x_and_teacher_rule1,
    )

    def link_z_with_x_and_teacher_rule2(model, c, t, teacher):
        return model.z[c, t, teacher] <= model.teacher_assignment[c, teacher]

    model.link_z_with_x_and_teacher2 = Constraint(
        model.Classes,
        model.TimeSlots,
        model.Teachers,
        rule=link_z_with_x_and_teacher_rule2,
    )

    def link_z_with_x_and_teacher_rule3(model, c, t, teacher):
        return (
            model.z[c, t, teacher]
            >= model.x[c, t] + model.teacher_assignment[c, teacher] - 1
        )

    model.link_z_with_x_and_teacher3 = Constraint(
        model.Classes,
        model.TimeSlots,
        model.Teachers,
        rule=link_z_with_x_and_teacher_rule3,
    )

    # 5. No two classes that share a common classroom can be in the same time slot
    def room_conflict_rule(model, t, room):
        return sum(model.w[c, t, room] for c in model.Classes) <= 1

    model.room_conflict = Constraint(
        model.TimeSlots, model.Classrooms, rule=room_conflict_rule
    )

    # Link w with x and room_assignment
    def link_w_with_x_and_room_rule1(model, c, t, room):
        return model.w[c, t, room] <= model.x[c, t]

    model.link_w_with_x_and_room1 = Constraint(
        model.Classes,
        model.TimeSlots,
        model.Classrooms,
        rule=link_w_with_x_and_room_rule1,
    )

    def link_w_with_x_and_room_rule2(model, c, t, room):
        return model.w[c, t, room] <= model.room_assignment[c, room]

    model.link_w_with_x_and_room2 = Constraint(
        model.Classes,
        model.TimeSlots,
        model.Classrooms,
        rule=link_w_with_x_and_room_rule2,
    )

    def link_w_with_x_and_room_rule3(model, c, t, room):
        return model.w[c, t, room] >= model.x[c, t] + model.room_assignment[c, room] - 1

    model.link_w_with_x_and_room3 = Constraint(
        model.Classes,
        model.TimeSlots,
        model.Classrooms,
        rule=link_w_with_x_and_room_rule3,
    )

    # Objective: Minimize the number of time slots used
    # This objective function aims to schedule classes as early as possible
    def minimize_time_slots(model):
        return sum(model.x[c, t] * t for c in model.Classes for t in model.TimeSlots)

    model.objective = Objective(rule=minimize_time_slots, sense=minimize)

    # Solve the model
    solver = Highs()
    solver.highs_options["time_limit"] = 30.0  # Set 30-second time limit

    start_time = time.time()
    results = solver.solve(model)
    solve_time = time.time() - start_time

    # Prepare debug information
    debug_info = {
        "termination_condition": results.termination_condition,
        "solve_time": solve_time,
        "objective_value": None,
        "best_objective_bound": None,
        "best_feasible_objective": None,
    }

    debug_info["objective_value"] = value(model.objective)
    debug_info["best_objective_bound"] = results.best_objective_bound
    debug_info["best_feasible_objective"] = results.best_feasible_objective

    logger.debug("Solver results: {}", debug_info)

    # Display results in a terminal table grid
    room_table = PrettyTable()
    room_names = sorted(list(model.Classrooms))
    room_table.field_names = ["Time Slot"] + room_names

    # Populate the table with class schedules
    for t in model.TimeSlots:
        row = [t]
        for room in room_names:
            room_class = ""
            for c in model.Classes:
                if (
                    model.x[c, t].value == 1
                    and model.room_assignment[c, room].value == 1
                ):
                    room_class = c
                    break
            row.append(room_class)
        room_table.add_row(row)

    teacher_table = PrettyTable()
    teacher_table.field_names = ["Time Slot"] + room_names

    # Populate the table with teacher schedules
    for t in model.TimeSlots:
        row = [t]
        for room in room_names:
            room_teacher = ""
            for c in model.Classes:
                if (
                    model.x[c, t].value == 1
                    and model.room_assignment[c, room].value == 1
                ):
                    room_teacher = [
                        teacher
                        for teacher in model.Teachers
                        if model.teacher_assignment[c, teacher].value == 1
                    ][0]
                    break
            row.append(room_teacher)
        teacher_table.add_row(row)

    logger.info("Room schedule:\n{}", room_table.get_string())
    logger.info("Teacher schedule:\n{}", teacher_table.get_string())

    return {
        "room_table": room_table.get_string(),
        "teacher_table": teacher_table.get_string(),
        "debug_info": debug_info,
    }


# Example usage with the updated data structure:
if __name__ == "__main__":
    data = {
        "Classes": ["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
        "TimeSlots": ["T1", "T2", "T3", "T4", "T5", "T6", "T7"],
        "Teachers": ["Teacher1", "Teacher2", "Teacher3", "Teacher4", "Teacher5"],
        "Classrooms": ["Room1", "Room2", "Room3", "Room4"],
    }

    # Call the function with the data
    res = create_and_solve_timetable_model(data)
    logger.info("Teacher table:\n{}", res["teacher_table"])
    logger.debug("Full results: {}", res)
