# ALWAYS MODIFY THIS CODE IN COMPLETE FORM, NEVER WRITE PLACEHOLDERS.
# THIS CODE MUST BE RUN DIRECTLY AS WRITTEN.

from pyomo.environ import *
from prettytable import PrettyTable
from pydantic import BaseModel
from typing import List, Optional, Literal
from pyomo.contrib.appsi.solvers import Highs

# This is a test comment to demonstrate patching

# Update the literal types to include C26
ClassLiteral = Literal[
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
    "C22",
    "C23",
    "C24",
    "C25",
    "C26",
]
TimeSlotLiteral = Literal["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]
TeacherLiteral = Literal["Teacher1", "Teacher2", "Teacher3", "Teacher4", "Teacher5"]
ClassroomLiteral = Literal["Room1", "Room2", "Room3", "Room4"]


class ForcedAssignment(BaseModel):
    classId: ClassLiteral
    timeslot: TimeSlotLiteral


class Constraints(BaseModel):
    OneTimeSlotPerClass: bool
    TeacherConflict: bool
    RoomConflict: bool


class TimetableInput(BaseModel):
    Classes: List[ClassLiteral]
    TimeSlots: List[TimeSlotLiteral]
    Teachers: List[TeacherLiteral]
    Classrooms: List[ClassroomLiteral]
    Constraints: Constraints
    ForcedAssignments: Optional[List[ForcedAssignment]] = None


class TimetableInputSchema(BaseModel):
    input: TimetableInput


# Create a model
model = ConcreteModel()


def create_and_solve_timetable_model(data):
    # IMPORTANT: TO MODIFY ENSURE YOU WRITE THE MODEL MATHEMATICALLY, MAKE CHANGES MATHEMATICALLY, ENSURE THE MODEL IS EFFICIENT MATHEMATICALLY THEN IMPLEMENT.
    # ALWAYS WRITE THE CURRENT MATHEMATICAL MODEL HERE:
    # Mathematical Model:
    # Sets:
    # C: Set of classes
    # T: Set of time slots
    # E: Set of teachers
    # R: Set of classrooms

    # Decision Variables:
    # x[c,t,e,r] = 1 if class c is assigned to time slot t, teacher e, and room r, 0 otherwise
    # y[e,t] = 1 if teacher e is teaching in time slot t, 0 otherwise
    # first[e]: First time slot for teacher e
    # last[e]: Last time slot for teacher e

    # Objective:
    # Minimize sum(last[e] - first[e] for e in E) + M * sum(1 - sum(x[c,t,e,r] for t in T, e in E, r in R) for c in C)
    # Where M is a large constant to prioritize assigning all classes

    # Constraints:
    # 1. Each class must be assigned exactly one time slot, teacher, and room:
    #    sum(x[c,t,e,r] for t in T, e in E, r in R) = 1 for all c in C
    # 2. No teacher conflicts:
    #    sum(x[c,t,e,r] for c in C, r in R) <= 1 for all t in T, e in E
    # 3. No room conflicts:
    #    sum(x[c,t,e,r] for c in C, e in E) <= 1 for all t in T, r in R
    # 4. Linking x and y:
    #    sum(x[c,t,e,r] for c in C, r in R) = y[e,t] for all e in E, t in T
    # 5. Defining first and last time slots for each teacher:
    #    first[e] <= t + (1 - y[e,t]) * |T| for all e in E, t in T
    #    last[e] >= t - (1 - y[e,t]) * |T| for all e in E, t in T
    # 6. Forced assignments (if any):
    #    x[c,t,e,r] = 1 for specified (c,t) pairs, for some e in E, r in R
    # 7. Ensure all classes are assigned:
    #    sum(x[c,t,e,r] for t in T, e in E, r in R) = 1 for all c in C
    # ----------------------------------------
    # ENSURE THE ABOVE MODEL AND THE CODE ARE COMPLETELY CONSISTENT.

    try:
        # Create a model
        model = ConcreteModel()

        # Sets
        model.Classes = Set(initialize=data["Classes"])
        model.TimeSlots = Set(
            initialize=list(range(1, len(data["TimeSlots"]) + 1))
        )  # Use indices
        model.Teachers = Set(initialize=data["Teachers"])
        model.Classrooms = Set(initialize=data["Classrooms"])

        # Decision Variables
        model.x = Var(
            model.Classes,
            model.TimeSlots,
            model.Teachers,
            model.Classrooms,
            domain=Binary,
        )
        model.y = Var(model.Teachers, model.TimeSlots, domain=Binary)
        # model.first = Var(model.Teachers, within=NonNegativeIntegers)
        # model.last = Var(model.Teachers, within=NonNegativeIntegers)

        # Simplified objective function for debugging
        def simplified_objective_rule(model):
            return sum(model.y[e, t] for e in model.Teachers for t in model.TimeSlots)

        model.objective = Objective(rule=simplified_objective_rule, sense=minimize)

        # Constraints
        # 1. Each class must be assigned exactly one time slot, teacher, and room
        def one_assignment_per_class_rule(model, c):
            return (
                sum(
                    model.x[c, t, e, r]
                    for t in model.TimeSlots
                    for e in model.Teachers
                    for r in model.Classrooms
                )
                == 1
            )

        model.one_assignment_per_class = Constraint(
            model.Classes, rule=one_assignment_per_class_rule
        )

        # 2. No teacher conflicts
        if data["Constraints"]["TeacherConflict"]:

            def teacher_conflict_rule(model, t, e):
                return (
                    sum(
                        model.x[c, t, e, r]
                        for c in model.Classes
                        for r in model.Classrooms
                    )
                    <= 1
                )

            model.teacher_conflict = Constraint(
                model.TimeSlots, model.Teachers, rule=teacher_conflict_rule
            )

        # 3. No room conflicts
        if data["Constraints"]["RoomConflict"]:

            def room_conflict_rule(model, t, r):
                return (
                    sum(
                        model.x[c, t, e, r]
                        for c in model.Classes
                        for e in model.Teachers
                    )
                    <= 1
                )

            model.room_conflict = Constraint(
                model.TimeSlots, model.Classrooms, rule=room_conflict_rule
            )

        # 4. Link x and y
        def link_x_y_rule(model, e, t):
            return (
                sum(
                    model.x[c, t, e, r] for c in model.Classes for r in model.Classrooms
                )
                == model.y[e, t]
            )

        model.link_x_y = Constraint(model.Teachers, model.TimeSlots, rule=link_x_y_rule)

        # Simplified constraint for debugging
        def simplified_rule(model):
            return sum(model.x[c, t, e, r] for c in model.Classes for t in model.TimeSlots for e in model.Teachers for r in model.Classrooms) >= 1

        model.simplified_constraint = Constraint(rule=simplified_rule)

        # 5. Ensure first and last slots are correctly identified
        # def min_first_time_slot_rule(model, e, t):
        #     return model.first[e] <= t + (1 - model.y[e, t]) * len(model.TimeSlots)

        # model.min_first_time_slot = Constraint(
        #     model.Teachers, model.TimeSlots, rule=min_first_time_slot_rule
        # )

        # def max_last_time_slot_rule(model, e, t):
        #     return model.last[e] >= t - (1 - model.y[e, t]) * len(model.TimeSlots)

        # model.max_last_time_slot = Constraint(
        #     model.Teachers, model.TimeSlots, rule=max_last_time_slot_rule
        # )

        # 6. Force specific classes into specific time slots
        if "ForcedAssignments" in data and data["ForcedAssignments"]:

            def force_assignment_rule(model, c, t):
                forced_assignments = {
                    entry["classId"]: entry["timeslot"]
                    for entry in data["ForcedAssignments"]
                }
                if c in forced_assignments and forced_assignments[c] == t:
                    return (
                        sum(
                            model.x[c, t, e, r]
                            for e in model.Teachers
                            for r in model.Classrooms
                        )
                        == 1
                    )
                else:
                    return Constraint.Skip

            model.force_assignment = Constraint(
                model.Classes, model.TimeSlots, rule=force_assignment_rule
            )

        # 7. Ensure all classes are assigned
        def all_classes_assigned_rule(model, c):
            return (
                sum(
                    model.x[c, t, e, r]
                    for t in model.TimeSlots
                    for e in model.Teachers
                    for r in model.Classrooms
                )
                == 1
            )

        model.all_classes_assigned = Constraint(
            model.Classes, rule=all_classes_assigned_rule
        )

        # Add transition tracking variables
        model.transition = Var(
            model.Teachers, model.TimeSlots, model.TimeSlots, domain=Binary
        )

        # Constraint to track transitions between different rooms for each teacher
        def transition_tracking_rule(model, e, t1, t2):
            if t1 < t2 and t2 == t1 + 1:
                return model.transition[e, t1, t2] == (
                    sum(
                        model.x[c, t1, e, r1]
                        for c in model.Classes
                        for r1 in model.Classrooms
                    )
                    * sum(
                        model.x[c, t2, e, r2]
                        for c in model.Classes
                        for r1 in model.Classrooms
                        for r2 in model.Classrooms
                        if r1 != r2
                    )
                )
            else:
                return Constraint.Skip

        model.transition_tracking = Constraint(
            model.Teachers,
            model.TimeSlots,
            model.TimeSlots,
            rule=transition_tracking_rule,
        )

        # Objective: Minimize the cost based on teachers' time at school and minimize teacher transitions between rooms
        def minimize_teacher_cost_and_transitions(model):
            M = 10000  # Large constant to prioritize assigning all classes
            transition_cost = 10  # Cost for teacher transition between rooms
            time_at_school_cost = sum(
                (model.last[e] - model.first[e] + 1) * 100 for e in model.Teachers
            )
            unassigned_classes_cost = M * sum(
                1
                - sum(
                    model.x[c, t, e, r]
                    for t in model.TimeSlots
                    for e in model.Teachers
                    for r in model.Classrooms
                )
                for c in model.Classes
            )
            transition_penalty = sum(
                transition_cost * model.transition[e, t1, t2]
                for e in model.Teachers
                for t1 in model.TimeSlots
                for t2 in model.TimeSlots
                if t1 < t2
            )
            return time_at_school_cost + unassigned_classes_cost + transition_penalty

        model.objective = Objective(
            rule=minimize_teacher_cost_and_transitions, sense=minimize
        )

        # Constraint: No teacher should have more than 6 hours of workload in a day
        def max_teacher_workload_rule(model, e):
            return sum(model.y[e, t] for t in model.TimeSlots) <= 6

        model.max_teacher_workload = Constraint(
            model.Teachers, rule=max_teacher_workload_rule
        )

        # Solve the model
        solver = Highs()
        res = solver.solve(model)

        # Check if the solution is optimal and feasible
        print("Termination Condition:", res.termination_condition)

        # Feasibility check
        for c in model.Classes:
            assigned = sum(
                model.x[c, t, e, r].value
                for t in model.TimeSlots
                for e in model.Teachers
                for r in model.Classrooms
            )
            if (
                assigned < 0.99
            ):  # Use 0.99 as threshold due to potential floating-point issues
                return {"error": f"Class {c} is not assigned (assigned = {assigned})"}

        # Calculate the total cost
        total_cost = model.objective()

        # Initialize debug_info
        debug_info = {}

        # Helper function to calculate teacher hours
        def calculate_teacher_hours(model, teacher):
            return sum(model.y[teacher, t].value for t in model.TimeSlots)

        # Add teacher hours to the debug info
        debug_info["teacher_hours"] = {
            e: calculate_teacher_hours(model, e) for e in model.Teachers
        }

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
                        sum(model.x[c, t, e, room].value for e in model.Teachers) > 0.5
                    ):  # Use 0.5 as threshold due to potential floating-point issues
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
                    for e in model.Teachers:
                        if (
                            model.x[c, t, e, room].value > 0.5
                        ):  # Use 0.5 as threshold due to potential floating-point issues
                            room_teacher = e
                            break
                    if room_teacher:
                        break
                row.append(room_teacher)
            teacher_table.add_row(row)

        # Update debugging information
        debug_info.update(
            {
                "model_stats": {
                    "num_variables": model.nvariables(),
                    "num_constraints": model.nconstraints(),
                    "num_objectives": model.nobjectives(),
                },
                "sets": {
                    "Classes": len(model.Classes),
                    "TimeSlots": len(model.TimeSlots),
                    "Teachers": len(model.Teachers),
                    "Classrooms": len(model.Classrooms),
                },
                "variable_stats": {
                    "x": sum(
                        model.x[c, t, e, r].value
                        for c in model.Classes
                        for t in model.TimeSlots
                        for e in model.Teachers
                        for r in model.Classrooms
                    ),
                    "y": sum(
                        model.y[e, t].value
                        for e in model.Teachers
                        for t in model.TimeSlots
                    ),
                },
                "constraint_stats": {
                    "one_assignment_per_class": [
                        model.one_assignment_per_class[c].body() for c in model.Classes
                    ],
                    "teacher_conflict": (
                        [
                            model.teacher_conflict[t, e].body()
                            for t in model.TimeSlots
                            for e in model.Teachers
                        ]
                        if hasattr(model, "teacher_conflict")
                        else "Not applied"
                    ),
                    "room_conflict": (
                        [
                            model.room_conflict[t, r].body()
                            for t in model.TimeSlots
                            for r in model.Classrooms
                        ]
                        if hasattr(model, "room_conflict")
                        else "Not applied"
                    ),
                },
                "objective_value": model.objective(),
                "solver_info": {
                    "termination_condition": res.termination_condition,
                },
            }
        )

        # Calculate actual teacher hours at school
        teacher_hours_at_school = {
            e: (
                (model.last[e].value - model.first[e].value + 1)
                if model.last[e].value > 0
                else 0
            )
            for e in model.Teachers
        }
        total_cost = sum(teacher_hours_at_school.values()) * 100

        debug_info["teacher_hours_at_school"] = teacher_hours_at_school
        debug_info["first_slot"] = {e: model.first[e].value for e in model.Teachers}
        debug_info["last_slot"] = {e: model.last[e].value for e in model.Teachers}

        return {
            "room_table": room_table.get_string(),
            "teacher_table": teacher_table.get_string(),
            "total_cost": total_cost,
            "debug_info": debug_info,
        }
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}


# Example usage with the updated data structure:
if __name__ == "__main__":
    data = {
        "Classes": ["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
        "TimeSlots": ["T1", "T2", "T3", "T4", "T5", "T6", "T7"],
        "Teachers": ["Teacher1", "Teacher2", "Teacher3", "Teacher4", "Teacher5"],
        "Classrooms": ["Room1", "Room2", "Room3", "Room4"],
        "Constraints": {
            "OneTimeSlotPerClass": True,
            "TeacherConflict": True,
            "RoomConflict": True,
        },
        "ForcedAssignments": [
            # {"classId": "C1", "timeslot": "T1"},  # Force C1 to be in T1
            # {"classId": "C2", "timeslot": "T2"},  # Force C2 to be in T2
        ],
    }

    # Call the function with the data
    res = create_and_solve_timetable_model(data)
    print(res)
    print(res["teacher_table"])

    # This is another test comment to demonstrate patching
    print("Timetable optimization complete!")