from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import importlib.util
import sys

import model

# set_debug(True)
# set_verbose(True)

memory = MemorySaver()


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


# Tool to modify the model code
@tool
def write_model(new_code: str) -> str:
    """Write the timetable optimization model code."""
    try:
        # Parse the new code to check for syntax errors
        # ast.parse(new_code)

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
    """Optimise the timetable"""
    try:
        import model as model_run

        data = input.model_dump()
        return model_run.create_and_solve_timetable_model(data)
    except Exception as e:
        return f"Error optimizing the timetable: {str(e)}. input was {input}"


# Now bind these functions to ToolNodes
tools = [read_model, write_model, time_table_optimiser]
tool_node = ToolNode(tools)

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
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
