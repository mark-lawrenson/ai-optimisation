import re
from typing import Annotated, List, Literal
import tiktoken
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import importlib.util
import sys
import model
import ast
from thefuzz import fuzz
from loguru import logger

# Configure loguru
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("file_{time}.log", rotation="500 MB", level="DEBUG")

memory = MemorySaver()
rate_limiter = InMemoryRateLimiter(requests_per_second=0.5)

MAX_TOKENS = 10000  # Truncating history to this tokens

# TODO: Use claude prompt caching? Keep trying to improve token usage.

system_prompt = """
You are an assistant specializing in linear optimization models for timetable scheduling. Your role is to help users interact with and modify a Mixed Integer Linear Programming (MILP) model for timetable optimization.

Key points to remember:
1. The model must always remain linear. Do not introduce any nonlinear elements under any circumstances.
2. Only use linear constraints and objective functions.
3. When suggesting changes, first describe the mathematical modifications, then implement them in code.
4. Always verify that your suggestions maintain linearity in the model.
5. If a user request would result in a nonlinear model, explain why it's not possible and suggest linear alternatives if available.

Common linear operations:
- Addition and subtraction of variables
- Multiplication of variables by constants
- Sum of variables (e.g., sum(x[i] for i in range(n)))
- Linear equalities and inequalities (e.g., x + y <= 5)

Forbidden nonlinear operations:
- Multiplication of variables (e.g., x * y)
- Division by variables
- Exponential or logarithmic functions
- Absolute value functions
- Quadratic or higher-order polynomials

When modifying the model:
1. Clearly state the proposed changes in mathematical notation.
2. Explain how these changes maintain linearity.
3. READ THE MODEL CODE TO ENSURE THAT YOUR CHANGES ARE VALID
4. Provide the diff to implement the changes to patch_model
5. Double-check that all constraints and objectives remain linear.
6. If you're unsure about the linearity of a proposed change, err on the side of caution and do not implement it.

Remember: Maintaining linearity is crucial. Always prioritize this constraint in your suggestions and implementations. If you cannot find a linear way to implement a requested feature, explain the limitations and suggest alternative approaches that maintain linearity.
"""


class State(TypedDict):
    messages: Annotated[list, add_messages]


def num_tokens_from_messages(messages, model="gpt-4o"):
    """Return the number of tokens used by a list of messages."""
    # NOTE: This uses tiktoken which is for openai models, might be good enough?
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += (
            4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        )
        num_tokens += len(encoding.encode(str(message.content)))
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


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

            # Handle indentation
            def get_min_leading_spaces(lines):
                return min(
                    (len(line) - len(line.lstrip()) for line in lines), default=0
                )

            min_leading_spaces_original = get_min_leading_spaces(
                lines[best_match : best_match + len(context)]
            )
            min_leading_spaces_changes = get_min_leading_spaces(changes)

            # Add indentation to changes
            additional_indentation = max(
                0, min_leading_spaces_original - min_leading_spaces_changes
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
                    f"Error: Couldn't find a match for context:\n{' '.join(context)}"
                )
        else:
            result.append(line)
        i += 1

    result.extend(lines)
    return "\n".join(result)


def apply_context_patches(original_code, patches):
    combined_patch_lines = patches.splitlines()
    if not combined_patch_lines[0].strip().startswith("<<<"):
        raise PatchingError("Error: Malformed Patch, did not start with <<<")

    split_patchlines = []
    current_patch = []

    for line in combined_patch_lines:
        if line.strip().startswith("<<<"):
            if current_patch:
                split_patchlines.append(current_patch)
            current_patch = [line]
        elif current_patch:
            current_patch.append(line)

    if current_patch:
        split_patchlines.append(current_patch)

    patched_code = original_code
    for i, patch_lines in enumerate(split_patchlines):
        print(f"Applying patch {i + 1}")
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

        logger.debug("Original code:\n{}", original_code)
        logger.debug("Patch:\n{}", patch)

        new_code = apply_context_patches(original_code, patch)

        logger.debug("New code after patching:\n{}", new_code)

        # Check for nonlinearity
        nonlinear = check_for_nonlinearity(new_code)
        logger.debug("Nonlinearity check result: {}", nonlinear)

        if nonlinear:
            return "Error: The proposed changes introduce nonlinearity into the model. Please revise the changes to maintain linearity."

        # Parse the new code to check for syntax errors
        ast.parse(new_code)

        # Write the new code to model.py
        with open("model.py", "w") as f:
            f.write(new_code)

        # Reload the module
        if "model" in sys.modules:
            logger.debug("Model module found in sys.modules, reloading it.")
            del sys.modules["model"]
        spec = importlib.util.spec_from_file_location("model", "model.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["model"] = module
        spec.loader.exec_module(module)

        return "Model successfully patched and reloaded."
    except SyntaxError as e:
        logger.error("Syntax error in the new code: {}", str(e))
        return f"Syntax error in the new code: {str(e)}"
    except PatchingError as e:
        logger.error("Error with the provided patch data: {}", str(e))
        return f"There was an error with the provided patch data: {str(e)}"
    except Exception as e:
        logger.error("Error modifying the model: {}", str(e))
        return f"Error modifying the model: {str(e)}"

def check_for_nonlinearity(code: str) -> bool:
    """Check if the given code introduces nonlinearity into the Pyomo model."""
    # This implementation focuses on Pyomo-specific nonlinear operations
    nonlinear_patterns = [
        r'model\.\w+(?!\.value)(?:\[[\w\s,\-+]*\])?\s*\*\s*model\.\w+(?!\.value)(?:\[[\w\s,\-+]*\])?',  # Multiplication of Pyomo variables
        r'model\.\w+(?!\.value)(?:\[[\w\s,\-+]*\])?\s*/\s*model\.\w+(?!\.value)(?:\[[\w\s,\-+]*\])?',  # Division by Pyomo variables
        r'exp\(\s*model\.\w+(?!\.value)(?:\[[\w\s,\-+]*\])?\s*\)',  # Exponential function with Pyomo variable
        r'log\(\s*model\.\w+(?!\.value)(?:\[[\w\s,\-+]*\])?\s*\)',  # Logarithmic function with Pyomo variable
        r'sqrt\(\s*model\.\w+(?!\.value)(?:\[[\w\s,\-+]*\])?\s*\)',  # Square root function with Pyomo variable
        r'abs\(\s*model\.\w+(?!\.value)(?:\[[\w\s,\-+]*\])?\s*\)',  # Absolute value function with Pyomo variable
        r'model\.\w+(?!\.value)(?:\[[\w\s,\-+]*\])?\s*\*\*\s*\d+',  # Power functions with Pyomo variable (e.g., model.x**2)
        r'sin\(\s*model\.\w+(?!\.value)(?:\[[\w\s,\-+]*\])?\s*\)',  # Trigonometric functions with Pyomo variable
        r'cos\(\s*model\.\w+(?!\.value)(?:\[[\w\s,\-+]*\])?\s*\)',
        r'tan\(\s*model\.\w+(?!\.value)(?:\[[\w\s,\-+]*\])?\s*\)',
    ]
    
    # Remove comments and import statements
    code_lines = [line.split('#')[0] for line in code.split('\n') if not line.strip().startswith(('import', 'from'))]
    cleaned_code = ' '.join(code_lines)
    
    for pattern in nonlinear_patterns:
        matches = re.finditer(pattern, cleaned_code)
        for match in matches:
            logger.debug("Nonlinearity detected: {} matches pattern {}", match.group(0), pattern)
            return True
    
    # Additional check for multiplication of indexed variables
    indexed_var_multiplication = r'model\.\w+(?:\[[\w\s,\-+]*\])?\s*\*\s*model\.\w+(?:\[[\w\s,\-+]*\])?'
    if re.search(indexed_var_multiplication, cleaned_code):
        logger.debug("Nonlinearity detected: Multiplication of indexed variables")
        return True
    
    logger.debug("No nonlinearity detected")
    return False


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
    max_tokens_to_sample=4096,
    rate_limiter=rate_limiter,
)
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    messages = state["messages"]
    # print("messages")
    # for msg in messages:
    #     print(msg.type, msg.name)
    # Always keep the system message
    kept_messages = [messages[0]]

    # Find the last read_model message
    last_read_model_index = None
    for i in range(len(messages) - 1, 0, -1):
        if messages[i].name == "read_model":
            last_read_model_index = i - 1  # Find the tool usage message
            break

    # Add messages from the start, skipping previous read_model pairs
    i = 1
    while i < len(messages) - 1:
        if messages[i].type == "ai" and messages[i + 1].type == "tool":
            if messages[i + 1].name == "read_model" and i != last_read_model_index:
                i += 2  # Skip this read_model pair
            else:
                kept_messages.extend(
                    messages[i : i + 2]
                )  # Keep the ai tool use and tool_result pair
                i += 2
        else:
            kept_messages.append(messages[i])
            i += 1
    if messages[-1].type != "tool":
        kept_messages.append(
            messages[-1]
        )  # Append the last message if it's not a tool use
    # print("kept_messages after read skipping")
    # for msg in kept_messages:
    #     print(msg.type, msg.name)
    # Count tokens and truncate if necessary
    message_index_to_drop = 2
    while num_tokens_from_messages(kept_messages) > MAX_TOKENS:
        if (
            len(kept_messages) > 6
        ):  # Keep system message, first user message last read_model pair (if any), and latest 2 messages
            # TODO: Improve this so that it drops old user messages while ensuring the first message after system is a user message
            if (
                kept_messages[message_index_to_drop + 1].type == "tool"
                and kept_messages[message_index_to_drop + 1].name == "read_model"
            ):
                logger.debug("NOT dropping read_model pair")
                message_index_to_drop += 2
            elif kept_messages[message_index_to_drop + 1].type == "tool":
                kept_messages = (
                    kept_messages[:message_index_to_drop]
                    + kept_messages[message_index_to_drop + 2 :]
                )
                logger.debug("Dropping tool use pair")
            else:
                kept_messages.pop(message_index_to_drop)
                logger.debug("Dropping message")
        else:
            break
    logger.debug("Kept messages after truncation: {}", [(msg.type, msg.name) for msg in kept_messages])
    # Generate response
    try:
        response = llm_with_tools.invoke(kept_messages)
    except Exception as e:
        # TODO: Put handling of hitting anthropic rate limit here
        logger.error("Error invoking model: {}", str(e))
        raise e

    # Update state with the new response
    new_messages = messages + [response]

    return {"messages": new_messages}


def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"


system_message = {"role": "system", "content": system_prompt}

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    should_continue,
)
graph_builder.add_edge("tools", "chatbot")

graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=memory)
initial_state = {"messages": [system_message]}

if __name__ == "__main__":
    state = initial_state
    while True:
        config = {"configurable": {"thread_id": "1"}}
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        state["messages"].append({"role": "user", "content": user_input})
        for event in graph.stream(state, config=config):
            for key, value in event.items():
                if key == "chatbot":
                    print("Assistant:", value["messages"][-1].content)
                elif key == "tools":
                    if value["messages"][-1].name != "read_model":
                        print("Tool Result:", value["messages"][-1].content)
                    else:
                        print("Tool Result: Model read.")
        state = event["chatbot"]
def check_for_nonlinearity(code: str) -> bool:
    """Check if the given code introduces nonlinearity into the model."""
    # This is a basic implementation and may need to be expanded
    # to cover all possible cases of nonlinearity
    nonlinear_patterns = [
        r'\w+\s*\*\s*\w+',  # Multiplication of variables
        r'\w+\s*/\s*\w+',  # Division by variables
        r'exp\(', r'log\(', r'sqrt\(',  # Exponential, logarithmic, and square root functions
        r'abs\(',  # Absolute value function
        r'\w+\s*\*\*\s*\d+',  # Power functions (e.g., x**2)
    ]
    
    for pattern in nonlinear_patterns:
        if re.search(pattern, code):
            return True
    return False
