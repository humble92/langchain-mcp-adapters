# credit: https://www.youtube.com/watch?v=0f3fTTXqTps
# Working on Linux, WSL; does not work on Windows
# Setup:
#   uv venv
#   uv sync

import sys
import os

sys.path.insert(0, os.getcwd())


# Imports Python's built-in asyncio module, which is used to run asynchronous code using async/await.
import asyncio

# ClientSession: manages a session with an MCP-compliant tool or service.
# StdioServerParameters: used to specify how to start the tool (like a subprocess).
from mcp import ClientSession, StdioServerParameters

# connects to a tool over standard input/output.
from mcp.client.stdio import stdio_client

# Imports a helper function that loads tools that support the MCP protocol, making them compatible with LangChain.
from langchain_mcp_adapters.tools import load_mcp_tools

# Imports a function to create an agent that follows the ReAct pattern (Reasoning + Acting) from LangGraph.
# ReAct agents can use tools and think step-by-step.
from langgraph.prebuilt import create_react_agent

# Imports the OpenAI chat model interface from LangChain.
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")

# Prepares the parameters to launch a tool server via math_server.py using Python.
server_params = StdioServerParameters(
    command="python",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["C:/Users/daniel.hwang/study/ml/mcp/langchain-mcp-adapters/tests/servers/math_server.py"],
)


# Defines an asynchronous main function.
async def main():
    # Multi server
    # Get the absolute path to the server scripts
    from pathlib import Path
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_core.tools import BaseTool

    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")
    weather_server_path = os.path.join(current_dir, "servers/weather_server.py")
    # math_server_path = "C:/Users/daniel.hwang/study/ml/mcp/langchain-mcp-adapters/tests/servers/math_server.py"
    # weather_server_path = "C:/Users/daniel.hwang/study/ml/mcp/langchain-mcp-adapters/tests/servers/weather_server.py"

    try:
        async with MultiServerMCPClient(
            {
                "math": {
                    "command": "python",
                    "args": [math_server_path],
                    "transport": "stdio",
                },
                "weather": {
                    "command": "python",
                    "args": [weather_server_path],
                    "transport": "stdio",
                },
            }
        ) as client:
            # Check that we have tools from both servers
            all_tools = client.get_tools()

            # Should have 3 tools (add, multiply, get_weather)
            assert len(all_tools) == 3

            # Check that tools are BaseTool instances
            for tool in all_tools:
                assert isinstance(tool, BaseTool)

            # Verify tool names
            tool_names = {tool.name for tool in all_tools}
            assert tool_names == {"add", "multiply", "get_weather"}

            # Check math server tools
            math_tools = client.server_name_to_tools["math"]
            assert len(math_tools) == 2
            math_tool_names = {tool.name for tool in math_tools}
            assert math_tool_names == {"add", "multiply"}

            # Check weather server tools
            weather_tools = client.server_name_to_tools["weather"]
            assert len(weather_tools) == 1
            assert weather_tools[0].name == "get_weather"
            print(weather_tools[0])

            # Test that we can call a math tool
            add_tool = next(tool for tool in all_tools if tool.name == "add")
            result = await add_tool.ainvoke({"a": 2, "b": 3})
            assert result == "5"

            # Test that we can call a weather tool
            weather_tool = next(tool for tool in all_tools if tool.name == "get_weather")
            result = await weather_tool.ainvoke({"location": "London"})
            assert result == "It's always sunny in London"

            # Test the multiply tool
            multiply_tool = next(tool for tool in all_tools if tool.name == "multiply")
            result = await multiply_tool.ainvoke({"a": 4, "b": 5})
            assert result == "20"
    except Exception as e:
        print("Failed to connect to servers:", e)


# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
