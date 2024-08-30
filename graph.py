from typing import Annotated, Type
from typing_extensions import TypedDict
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool
from langchain_core.globals import set_debug, set_verbose
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
import importlib.util
import sys
import ast

import model

set_debug(False)
set_verbose(False)

memory = MemorySaver()


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


class ReadModelTool(BaseTool):
    name = "read_model"
    description = "Read the timetable optimization model code"

    def _run(self) -> str:
        """Modify the model code."""
        try:

            # Write the new code to model.py
            with open("model.py", "r") as f:
                code = f.read()

            return code
        except Exception as e:
            return f"Error reading the model: {str(e)}"


class ModifyModelTool(BaseTool):
    name = "modify_model"
    description = "Modify the timetable optimization model code"

    class ModifyModelInput(BaseModel):
        new_code: str

    args_schema: Type[BaseModel] = ModifyModelInput

    def _run(self, new_code: str) -> str:
        """Modify the model code."""
        try:
            # Parse the new code to check for syntax errors
            ast.parse(new_code)

            # Write the new code to model.py
            with open("model.py", "w") as f:
                f.write(new_code)

            # Reload the module
            spec = importlib.util.spec_from_file_location("model", "model.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules["model"] = module
            spec.loader.exec_module(module)

            return "Model successfully modified and reloaded."
        except SyntaxError as e:
            return f"Syntax error in the new code: {str(e)}"
        except Exception as e:
            return f"Error modifying the model: {str(e)}"


class TimetableOptimiserTool(BaseTool):
    name = "time_table_optimiser"
    description = "Optimise the timetable"
    args_schema: Type[BaseModel] = model.TimetableInputSchema

    def _run(self, input: model.TimetableInput) -> str:
        """Use the tool."""
        data = input.model_dump()
        return model.create_and_solve_timetable_model(data)


tools = [TimetableOptimiserTool(), ModifyModelTool(), ReadModelTool()]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
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
        for event in graph.stream({"messages": ("user", user_input)}, config=config):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)
