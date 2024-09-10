# ALWAYS MODIFY THIS CODE IN COMPLETE FORM, NEVER WRITE PLACEHOLDERS.
# THIS CODE MUST BE RUN DIRECTLY AS WRITTEN.

from pyomo.environ import *
from prettytable import PrettyTable
from pydantic import BaseModel
from typing import List, Optional, Literal
from pyomo.contrib.appsi.solvers import Highs

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
    # x[c,t] = 1 if class c is assigned to time slot t, 0 otherwise
    # y[e,t] = 1 if teacher e is teaching in time slot t, 0 otherwise
    # z[c,t,e] = 1 if class c is assigned to time slot t and teacher e, 0 otherwise
    # w[c,t,r] = 1 if class c is assigned to time slot t and classroom r, 0 otherwise
    # teacher_assignment[c,e] = 1 if teacher e is assigned to class c, 0 otherwise
    # room_assignment[c,r] = 1 if classroom r is assigned to class c, 0 otherwise
    # first[e]: First time slot for teacher e
    # last[e]: Last time slot for teacher e

    # Objective:
    # Minimize sum(last[e] - first[e] for e in E)

    # Constraints:
    # 1. Each class must be assigned exactly one time slot:
    #    sum(x[c,t] for t in T) = 1 for all c in C
    # 2. Each class must be assigned exactly one teacher:
    #    sum(teacher_assignment[c,e] for e in E) = 1 for all c in C
    # 3. Each class must be assigned exactly one room:
    #    sum(room_assignment[c,r] for r in R) = 1 for all c in C
    # 4. No teacher conflicts:
    #    sum(z[c,t,e] for c in C) <= 1 for all t in T, e in E
    # 5. No room conflicts:
    #    sum(w[c,t,r] for c in C) <= 1 for all t in T, r in R
    # 6. Linking constraints for z:
    #    z[c,t,e] <= x[c,t] for all c in C, t in T, e in E
    #    z[c,t,e] <= teacher_assignment[c,e] for all c in C, t in T, e in E
    #    z[c,t,e] >= x[c,t] + teacher_assignment[c,e] - 1 for all c in C, t in T, e in E
    # 7. Linking constraints for w:
    #    w[c,t,r] <= x[c,t] for all c in C, t in T, r in R
    #    w[c,t,r] <= room_assignment[c,r] for all c in C, t in T, r in R
    #    w[c,t,r] >= x[c,t] + room_assignment[c,r] - 1 for all c in C, t in T, r in R
    # 8. Linking x and y:
    #    sum(z[c,t,e] for c in C) >= y[e,t] for all e in E, t in T
    # 9. Defining first and last time slots for each teacher:
    #    first[e] <= t + (1 - y[e,t]) * |T| for all e in E, t in T
    #    last[e] >= t - (1 - y[e,t]) * |T| for all e in E, t in T
    # ----------------------------------------
    # ENSURE THE ABOVE MODEL AND THE CODE ARE COMPLETELY CONSISTENT.

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
    model.x = Var(model.Classes, model.TimeSlots, domain=Binary)
    model.teacher_assignment = Var(model.Classes, model.Teachers, domain=Binary)
    model.room_assignment = Var(model.Classes, model.Classrooms, domain=Binary)
    model.y = Var(model.Teachers, model.TimeSlots, domain=Binary)
    model.first = Var(model.Teachers, within=NonNegativeIntegers)
    model.last = Var(model.Teachers, within=NonNegativeIntegers)
    model.z = Var(model.Classes, model.TimeSlots, model.Teachers, domain=Binary)
    model.w = Var(model.Classes, model.TimeSlots, model.Classrooms, domain=Binary)
    model.v = Var(model.Classes, model.TimeSlots, model.Teachers, domain=Binary)

    # Constraints
    # 1. Each class must be assigned exactly one time slot
    if data["Constraints"]["OneTimeSlotPerClass"]:

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
    if data["Constraints"]["TeacherConflict"]:

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
    if data["Constraints"]["RoomConflict"]:

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

    # 6. Force specific classes into specific time slots
    if "ForcedAssignments" in data and data["ForcedAssignments"]:

        def force_assignment_rule(model, c, t):
            forced_assignments = {
                entry["classId"]: entry["timeslot"]
                for entry in data["ForcedAssignments"]
            }
            if c in forced_assignments and forced_assignments[c] == t:
                return model.x[c, t] == 1
            else:
                return Constraint.Skip

        model.force_assignment = Constraint(
            model.Classes, model.TimeSlots, rule=force_assignment_rule
        )

    # Link x and y: y[e, t] should be 1 if any class taught by teacher e is at time slot t
    def link_x_y_rule(model, e, t):
        return sum(model.v[c, t, e] for c in model.Classes) >= model.y[e, t]

    model.link_x_y = Constraint(model.Teachers, model.TimeSlots, rule=link_x_y_rule)

    # Link v with x and teacher_assignment
    def link_v_with_x_and_teacher_rule1(model, c, t, teacher):
        return model.v[c, t, teacher] <= model.x[c, t]

    model.link_v_with_x_and_teacher1 = Constraint(
        model.Classes,
        model.TimeSlots,
        model.Teachers,
        rule=link_v_with_x_and_teacher_rule1,
    )

    def link_v_with_x_and_teacher_rule2(model, c, t, teacher):
        return model.v[c, t, teacher] <= model.teacher_assignment[c, teacher]

    model.link_v_with_x_and_teacher2 = Constraint(
        model.Classes,
        model.TimeSlots,
        model.Teachers,
        rule=link_v_with_x_and_teacher_rule2,
    )

    def link_v_with_x_and_teacher_rule3(model, c, t, teacher):
        return (
            model.v[c, t, teacher]
            >= model.x[c, t] + model.teacher_assignment[c, teacher] - 1
        )

    model.link_v_with_x_and_teacher3 = Constraint(
        model.Classes,
        model.TimeSlots,
        model.Teachers,
        rule=link_v_with_x_and_teacher_rule3,
    )

    # Ensure first and last slots are correctly identified
    def min_first_time_slot_rule(model, e, t):
        return model.first[e] <= t + (1 - model.y[e, t]) * len(model.TimeSlots)

    model.min_first_time_slot = Constraint(
        model.Teachers, model.TimeSlots, rule=min_first_time_slot_rule
    )

    def max_last_time_slot_rule(model, e, t):
        return model.last[e] >= t - (1 - model.y[e, t]) * len(model.TimeSlots)

    model.max_last_time_slot = Constraint(
        model.Teachers, model.TimeSlots, rule=max_last_time_slot_rule
    )

    # Objective: Minimize the working span for each teacher
    def minimize_teacher_working_span(model):
        return sum(model.last[e] - model.first[e] for e in model.Teachers)

    model.objective = Objective(rule=minimize_teacher_working_span, sense=minimize)

    # Solve the model
    solver = Highs()
    solver.solve(model)

    # Calculate the total cost
    teacher_cost_per_hour = 100
    total_cost = sum(
        (model.last[e].value - model.first[e].value + 1) * teacher_cost_per_hour
        for e in model.Teachers
    )

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

    return {
        "room_table": room_table.get_string(),
        "teacher_table": teacher_table.get_string(),
        "total_cost": total_cost,
    }


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
    print(res["teacher_table"])
    print(res)
