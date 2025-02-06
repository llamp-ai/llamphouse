from llamphouse.core import Assistant
from llamphouse.core.context import Context
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import StateGraph, START, END
from typing import Optional, Annotated
from typing_extensions import TypedDict

# import the tools
from tools.get_coordinates_by_query import get_coordinates_by_query
from tools.get_temperature_from_lat_lon import get_temperature_from_lat_lon

def add_messages(messages: list[HumanMessage | AIMessage], data: list[HumanMessage | AIMessage]) -> list[HumanMessage | AIMessage]:
    return data # return the new list of messages

# Define the state of the graph
class State(TypedDict):
    messages: Annotated[list[HumanMessage | AIMessage], add_messages]
    latitude: float
    longitude: float
    user_location_name: Optional[float] = None

# LangGraph node for executing tools
class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        for tool_call in message.tool_calls:

            if tool_call["name"] == get_coordinates_by_query.name:
                tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )

                inputs["user_location_name"] = tool_call["args"].get("query", None)

                if "latitude" in tool_result and "longitude" in tool_result:
                    inputs["latitude"] = tool_result["latitude"]
                    inputs["longitude"] = tool_result["longitude"]

            elif tool_call["name"] == get_temperature_from_lat_lon.name:
                tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )
                
            inputs["messages"].append(
                ToolMessage(
                    content=tool_result,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

        return inputs

def route_tools(state):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

# Define the llamphouse assistant
class TemperatureAssistant(Assistant):
    def __init__(self, id: str):
        super().__init__(model="gpt-4o", id=id, instructions="I can help you with temperature queries.")

    def run(self, context: Context, *args, **kwargs):

        chat_messages = []
        for msg in reversed(context.messages):
            if msg.role == "user":
                chat_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                chat_messages.append(AIMessage(content=msg.content))
            elif msg.role == 'system':
                continue
            else:
                raise Exception(f"Unknown role encountered in chat history: {msg.role}.")

        # Define the available tools
        tools = [get_coordinates_by_query, get_temperature_from_lat_lon]

        llm = ChatOpenAI(temperature=0)

        def chatbot(state: State):
            llm_with_tools = llm.bind_tools(tools=[convert_to_openai_tool(tool) for tool in tools])
            response = llm_with_tools.invoke(state["messages"])
            return {
                "messages": state["messages"] + [response]
            }

        state = State(messages=chat_messages)

        # Define the graph structure 
        graph_builder = StateGraph(State)

        # Add the tools node to the graph
        tool_node = BasicToolNode(tools=tools)
        graph_builder.add_node("tools", tool_node)

        # Add the chatbot node to the graph
        graph_builder.add_node("chatbot", chatbot)

        # Add the start and end nodes to the graph
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)

        # The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
        # it is fine directly responding. This conditional routing defines the main agent loop.
        graph_builder.add_conditional_edges(
            "chatbot",
            route_tools,
            {"tools": "tools", END: END},
        )
        # Any time a tool is called, we return to the chatbot to decide the next step
        graph_builder.add_edge("tools", "chatbot")

        graph = graph_builder.compile()

        response = graph.invoke(state)

        last_assistant_message = [msg for msg in response['messages'] if type(msg) == AIMessage][-1]

        context.create_message(last_assistant_message.content)