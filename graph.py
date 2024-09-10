from typing import Annotated, List, Literal
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic

# from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import importlib.util
import sys
import model
import ast
from thefuzz import fuzz

# set_debug(True)
# set_verbose(True)

memory = MemorySaver()

# TODO: Use claude caching

system_prompt = """
You are an assistant that helps users interact with optimisation models.
In particular you have a timetable optimisation tool, as well as the capability to read and edit the code of this optimisation model.
When changing the model, you must first think about the changes you want to make to the mathematical model, summarise the new mathematical model, then implement the changes in code.
The mathematical model can only be MILP, it does not suport general expressions or nonlinearity at all.
When implementing changes in code ensure you patch the code using the current version of the code - reread the code if necessary.
"""


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


class PatchingError(ValueError):
    """Custom exception for invalid patching inputs."""

    pass


def find_best_match(lines: List[str], context: List[str], threshold: int = 80) -> int:
    """Find the best match for a context in the lines using a sliding window approach."""
    best_score = 0
    best_index = -1

    for i in range(len(lines) - len(context) + 1):
        window = lines[i : i + len(context)]
        score = sum(fuzz.ratio(a, b) for a, b in zip(window, context)) / len(context)
        if score > best_score:
            best_score = score
            best_index = i

    return best_index if best_score >= threshold else -1


def apply_context_patch(original: str, patch: str) -> str:
    """Apply a context-based patch to the original text."""
    lines = original.splitlines()
    patch_lines = patch.splitlines()
    result = []
    i = 0

    while i < len(patch_lines):
        line = patch_lines[i].strip()
        if line.startswith("<<<"):
            # Extract context and changes
            context = []
            changes = []
            i += 1
            try:
                while i < len(patch_lines) and not patch_lines[i].strip().startswith(
                    "---"
                ):
                    context.append(patch_lines[i])
                    i += 1
                i += 1  # Skip the '---' line
                while i < len(patch_lines) and not patch_lines[i].strip().startswith(
                    ">>>"
                ):
                    changes.append(patch_lines[i])
                    i += 1
                i += 1  # Skip the '>>>' line
            except IndexError:
                raise PatchingError(
                    f"Error: Malformed patch. Unexpected end of patch data at line {i}"
                )

            # Find the best match for the context
            best_match = find_best_match(lines, context)

            # deal with indentation
            # Figure out the least indented common across the best match
            min_leading_spaces_original = 999
            for l in lines[best_match : best_match + len(context)]:
                min_leading_spaces_original = min(
                    min_leading_spaces_original, len(l) - len(l.lstrip(" "))
                )

            # Figure out the least indented common across the best match
            min_leading_spaces_changes = 999
            for l in changes:
                min_leading_spaces_changes = min(
                    min_leading_spaces_changes, len(l) - len(l.lstrip(" "))
                )

            # Add indentation to changes
            additional_indentation = (
                min_leading_spaces_original - min_leading_spaces_changes
            )
            changes = [" " * additional_indentation + change for change in changes]

            if best_match != -1:
                # Apply the changes
                result.extend(lines[:best_match])
                result.extend(changes)
                result.extend(lines[best_match + len(context) :])
                lines = result
                result = []
            else:
                raise PatchingError(
                    f"Error: Couldn't find a match for context:\n{context}"
                )
        else:
            result.append(line)
        i += 1

    result.extend(lines)
    return "\n".join(result)


def apply_context_patches(original_code, patches):
    # Split the patches up and call apply_context_patch on each of them
    combined_patch_lines = patches.splitlines()
    if not combined_patch_lines[0].startswith("<<<"):
        raise PatchingError("Error: Malformed Patch, did not start with <<<")
    split_patchlines = list()
    for line in combined_patch_lines:
        if line.startswith("<<<"):
            split_patchlines.append([])
        split_patchlines[-1].append(line)
    patched_code = original_code
    for i, patch_lines in enumerate(split_patchlines):
        print(f"Patching patch {i}")
        patched_code = apply_context_patch(patched_code, "\n".join(patch_lines))
    return patched_code


@tool
def patch_model(patch: str) -> str:
    """Patch the timetable optimization model code using a context-based patch format.
    Please use the following format for each change
    ```
    <<<
    [complete lines to find in the code]
    ---
    [new code to replace the found lines entirely]
    >>>
    ```
    Make multiple changes by using the above format repeatedly in the input, prefer to make multiple smaller changes over one larger change.
    Keep the find and replace blocks as short as possible to implement the desired changes.
    DO NOT INCLUDE ANY PLACEHOLDERS IN THE NEW CODE!!!
    """
    try:
        with open("model.py", "r") as f:
            original_code = f.read()

        # Write the new code to model.py
        with open("model.patch", "w") as f:
            f.write(patch)

        new_code = apply_context_patches(original_code, patch)

        # Write the new code to model.py
        with open("model_patched.py", "w") as f:
            f.write(new_code)

        # Parse the new code to check for syntax errors
        ast.parse(new_code)

        # Write the new code to model.py
        with open("model.py", "w") as f:
            f.write(new_code)

        # Reload the module
        if "model" in sys.modules:
            print("DEBUG: Model module found in sys.modules, reloading it.")
            del sys.modules["model"]
        spec = importlib.util.spec_from_file_location("model", "model.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["model"] = module
        spec.loader.exec_module(module)

        return "Model successfully patched and reloaded."
    except SyntaxError as e:
        return f"Syntax error in the new code: {str(e)}"
    except PatchingError as e:
        return f"There was an error with the provided patch data: {str(e)}"
    except Exception as e:
        return f"Error modifying the model: {str(e)}"


# Tool to write the model code
@tool
def write_model(new_code: str) -> str:
    """Write the timetable optimization model code."""
    try:

        # Write the new code to model.py
        with open("model.py", "w") as f:
            f.write(new_code)

        # Reload the module
        if "model" in sys.modules:
            print("DEBUG: Model module found in sys.modules, reloading it.")
            # Remove the module from the system cache to ensure it's reloaded
            del sys.modules["model"]
        spec = importlib.util.spec_from_file_location("model", "model.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["model"] = module
        spec.loader.exec_module(module)

        return "Model successfully written and reloaded."
    except SyntaxError as e:
        return f"Syntax error in the new code: {str(e)}"
    except Exception as e:
        return f"Error modifying the model: {str(e)}"


# Tool to read the model code
@tool
def read_model() -> str:
    """Read the current timetable optimization model code."""
    try:
        with open("model.py", "r") as f:
            code = f.read()
        with open("model_read.py", "w") as f:
            f.write(code)
        return code
    except Exception as e:
        return f"Error reading the model: {str(e)}"


# Tool to optimize the timetable
@tool(args_schema=model.TimetableInputSchema)
def time_table_optimiser(input: model.TimetableInput) -> str:
    """Optimise the timetable
    When displaying output always include the pretty tables"""
    try:
        import model as model_run

        data = input.model_dump()
        return model_run.create_and_solve_timetable_model(data)
    except Exception as e:
        return f"Error optimizing the timetable: {str(e)}. input was {input}"


# Now bind these functions to ToolNodes
tools = [read_model, patch_model, time_table_optimiser]
tool_node = ToolNode(tools)

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    max_tokens_to_sample=8192,
    model_kwargs=dict(system=system_prompt),
)
# llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)


graph_builder.add_conditional_edges(
    "chatbot",
    should_continue,
)
graph_builder.add_edge("tools", "chatbot")

graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=memory)

if __name__ == "__main__":
    while True:
        config = {"configurable": {"thread_id": "1"}}
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        for event in graph.stream({"messages": [("user", user_input)]}, config=config):
            for key, value in event.items():
                if key == "chatbot":
                    print("Assistant:", value["messages"][-1].content)
                elif key == "tools":
                    if value["messages"][-1].name != "read_model":
                        print("Tool Result:", value["messages"][-1].content)
                    else:
                        print("Tool Result: Model read.")
