# Quickstart



## Install Dependencies

1. Install glpk solver
```bash
arch -arm64 brew install glpk
```

2. Create virtual environment and install pip packages

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


3. Run program
```bash
python graph.py
```


## Usage

As a user you may ask a question to the bot:
```
User: Are you working?
Assistant: Yes, I'm here and ready to assist you. How can I help you today?
```

You may ask it to optimise your timetable:
```
User: Please optimise my timetable
```

It will likely ask for details
```
Assistant: To help you optimize your timetable, I'll need the following information:

1. List of Classes (e.g., C1, C2, C3, etc.).
2. List of Time Slots (e.g., T1, T2, T3, etc.).
3. List of Teachers (e.g., Teacher1, Teacher2, Teacher3, etc.).
4. List of Classrooms (e.g., Room1, Room2, Room3, etc.).
5. Mapping of each Class to its Teacher.
6. Mapping of each Class to its Classroom.
7. Any constraints (e.g., one time slot per class, no teacher conflicts, no room conflicts).
8. Any forced assignments (if applicable).

Please provide the details above so I can proceed with optimizing your timetable.
```

You can ask it to provide dummy data:
```
User: Please create a dummy scenario
Assistant: 
Assistant: Here is the optimized timetable based on the dummy data provided:

+-----------+-------+-------+-------+-------+
| Time Slot | Room1 | Room2 | Room3 | Room4 |
+-----------+-------+-------+-------+-------+
|     T1    |   C1  |       |       |       |
|     T2    |       |       |       |       |
|     T3    |       |       |       |       |
|     T4    |       |       |       |       |
|     T5    |   C5  |   C2  |   C3  |   C4  |
+-----------+-------+-------+-------+-------+

+-----------+----------+----------+----------+----------+
| Time Slot |  Room1   |  Room2   |  Room3   |  Room4   |
+-----------+----------+----------+----------+----------+
|     T1    | Teacher1 |          |          |          |
|     T2    |          |          |          |          |
|     T3    |          |          |          |          |
|     T4    |          |          |          |          |
|     T5    | Teacher1 | Teacher2 | Teacher3 | Teacher4 |
+-----------+----------+----------+----------+----------+


This timetable satisfies the constraints:
- Only one time slot per class.
- No teacher has a conflict (i.e., no teacher is assigned to more than one class at the same time).
- No room has a conflict (i.e., no room is assigned to more than one class at the same time).
```

You can then make refinements:

```
User: please make c2 in t3


Assistant: 
Class Timetable:
+-----------+-------+-------+-------+-------+
| Time Slot | Room1 | Room2 | Room3 | Room4 |
+-----------+-------+-------+-------+-------+
|     T1    |   C1  |       |       |       |
|     T2    |       |       |       |       |
|     T3    |       |   C2  |       |       |
|     T4    |       |       |       |       |
|     T5    |   C5  |       |   C3  |   C4  |
+-----------+-------+-------+-------+-------+
+-----------+----------+----------+----------+----------+
| Time Slot |  Room1   |  Room2   |  Room3   |  Room4   |
+-----------+----------+----------+----------+----------+
|     T1    | Teacher1 |          |          |          |
|     T2    |          |          |          |          |
|     T3    |          | Teacher2 |          |          |
|     T4    |          |          |          |          |
|     T5    | Teacher1 |          | Teacher3 | Teacher4 |
+-----------+----------+----------+----------+----------+

This updated timetable still satisfies all constraints:
- Only one time slot per class.
- No teacher conflicts.
- No room conflicts.

```

And again

```
User: make c4 in t4
Class Timetable:
+-----------+-------+-------+-------+-------+
| Time Slot | Room1 | Room2 | Room3 | Room4 |
+-----------+-------+-------+-------+-------+
|     T1    |   C1  |       |       |       |
|     T2    |       |       |       |       |
|     T3    |       |   C2  |       |       |
|     T4    |       |       |       |   C4  |
|     T5    |   C5  |       |   C3  |       |
+-----------+-------+-------+-------+-------+
+-----------+----------+----------+----------+----------+
| Time Slot |  Room1   |  Room2   |  Room3   |  Room4   |
+-----------+----------+----------+----------+----------+
|     T1    | Teacher1 |          |          |          |
|     T2    |          |          |          |          |
|     T3    |          | Teacher2 |          |          |
|     T4    |          |          |          | Teacher4 |
|     T5    | Teacher1 |          | Teacher3 |          |
+-----------+----------+----------+----------+----------+


This updated timetable still satisfies all constraints:
- Only one time slot per class.
- No teacher conflicts.
- No room conflicts.
```