Function tools¶
What are function tools?¶
When out-of-the-box tools don't fully meet specific requirements, developers can create custom function tools. This allows for tailored functionality, such as connecting to proprietary databases or implementing unique algorithms.

For example, a function tool, "myfinancetool", might be a function that calculates a specific financial metric. ADK also supports long running functions, so if that calculation takes a while, the agent can continue working on other tasks.

ADK offers several ways to create functions tools, each suited to different levels of complexity and control:

Function Tool
Long Running Function Tool
Agents-as-a-Tool
1. Function Tool¶
Transforming a function into a tool is a straightforward way to integrate custom logic into your agents. In fact, when you assign a Python function to an agent’s tools list, the framework will automatically wrap it as a Function Tool for you. This approach offers flexibility and quick integration.

Parameters¶
Define your function parameters using standard JSON-serializable types (e.g., string, integer, list, dictionary). It's important to avoid setting default values for parameters, as the language model (LLM) does not currently support interpreting them.

Return Type¶
The preferred return type for a Python Function Tool is a dictionary. This allows you to structure the response with key-value pairs, providing context and clarity to the LLM. If your function returns a type other than a dictionary, the framework automatically wraps it into a dictionary with a single key named "result".

Strive to make your return values as descriptive as possible. For example, instead of returning a numeric error code, return a dictionary with an "error_message" key containing a human-readable explanation. Remember that the LLM, not a piece of code, needs to understand the result. As a best practice, include a "status" key in your return dictionary to indicate the overall outcome (e.g., "success", "error", "pending"), providing the LLM with a clear signal about the operation's state.

Docstring¶
The docstring of your function serves as the tool's description and is sent to the LLM. Therefore, a well-written and comprehensive docstring is crucial for the LLM to understand how to use the tool effectively. Clearly explain the purpose of the function, the meaning of its parameters, and the expected return values.

Example
This tool is a python function which obtains the Stock price of a given Stock ticker/ symbol.

Note: You need to pip install yfinance library before using this tool.


from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import yfinance as yf


APP_NAME = "stock_app"
USER_ID = "1234"
SESSION_ID = "session1234"

def get_stock_price(symbol: str):
    """
    Retrieves the current stock price for a given symbol.

    Args:
        symbol (str): The stock symbol (e.g., "AAPL", "GOOG").

    Returns:
        float: The current stock price, or None if an error occurs.
    """
    try:
        stock = yf.Ticker(symbol)
        historical_data = stock.history(period="1d")
        if not historical_data.empty:
            current_price = historical_data['Close'].iloc[-1]
            return current_price
        else:
            return None
    except Exception as e:
        print(f"Error retrieving stock price for {symbol}: {e}")
        return None


stock_price_agent = Agent(
    model='gemini-2.0-flash',
    name='stock_agent',
    instruction= 'You are an agent who retrieves stock prices. If a ticker symbol is provided, fetch the current price. If only a company name is given, first perform a Google search to find the correct ticker symbol before retrieving the stock price. If the provided ticker symbol is invalid or data cannot be retrieved, inform the user that the stock price could not be found.',
    description='This agent specializes in retrieving real-time stock prices. Given a stock ticker symbol (e.g., AAPL, GOOG, MSFT) or the stock name, use the tools and reliable data sources to provide the most up-to-date price.',
    tools=[get_stock_price], # You can add Python functions directly to the tools list; they will be automatically wrapped as FunctionTools.
)


# Session and Runner
session_service = InMemorySessionService()
session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=stock_price_agent, app_name=APP_NAME, session_service=session_service)


# Agent Interaction
def call_agent(query):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("Agent Response: ", final_response)

call_agent("stock price of GOOG")
The return value from this tool will be wrapped into a dictionary.


{"result": "$123"}
Best Practices¶
While you have considerable flexibility in defining your function, remember that simplicity enhances usability for the LLM. Consider these guidelines:

Fewer Parameters are Better: Minimize the number of parameters to reduce complexity.
Simple Data Types: Favor primitive data types like str and int over custom classes whenever possible.
Meaningful Names: The function's name and parameter names significantly influence how the LLM interprets and utilizes the tool. Choose names that clearly reflect the function's purpose and the meaning of its inputs. Avoid generic names like do_stuff().
2. Long Running Function Tool¶
Designed for tasks that require a significant amount of processing time without blocking the agent's execution. This tool is a subclass of FunctionTool.

When using a LongRunningFunctionTool, your Python function can initiate the long-running operation and optionally return an initial result** (e.g. the long-running operation id). Once a long running function tool is invoked the agent runner will pause the agent run and let the agent client to decide whether to continue or wait until the long-running operation finishes. The agent client can query the progress of the long-running operation and send back an intermediate or final response. The agent can then continue with other tasks. An example is the human-in-the-loop scenario where the agent needs human approval before proceeding with a task.

How it Works¶
You wrap a Python function with LongRunningFunctionTool.

Initiation: When the LLM calls the tool, your python function starts the long-running operation.

Initial Updates: Your function should optionally return an initial result (e.g. the long-running operaiton id). The ADK framework takes the result and sends it back to the LLM packaged within a FunctionResponse. This allows the LLM to inform the user (e.g., status, percentage complete, messages). And then the agent run is ended / paused.

Continue or Wait: After each agent run is completed. Agent client can query the progress of the long-running operation and decide whether to continue the agent run with an intermediate response (to update the progress) or wait until a final response is retrieved. Agent client should send the intermediate or final response back to the agent for the next run.

Framework Handling: The ADK framework manages the execution. It sends the intermediate or final FunctionResponse sent by agent client to the LLM to generate a user friendly message.

Creating the Tool¶
Define your tool function and wrap it using the LongRunningFunctionTool class:


from google.adk.tools import LongRunningFunctionTool

# Define your long running function (see example below)
def ask_for_approval(
    purpose: str, amount: float, tool_context: ToolContext
) -> dict[str, Any]:
  """Ask for approval for the reimbursement."""
  # create a ticket for the approval
  # Send a notification to the approver with the link of the ticket
  return {'status': 'pending', 'approver': 'Sean Zhou', 'purpose' : purpose, 'amount': amount, 'ticket-id': 'approval-ticket-1'}

# Wrap the function
approve_tool = LongRunningFunctionTool(func=ask_for_approval)
Intermediate / Final result Updates¶
Agent client received an event with long running function calls and check the status of the ticket. Then Agent client can send the intermediate or final response back to update the progress. The framework packages this value (even if it's None) into the content of the FunctionResponse sent back to the LLM.


# runner = Runner(...)
# session = session_service.create_session(...)
# content = types.Content(...) # User's initial query

def get_long_running_function_call(event: Event) -> types.FunctionCall:
    # Get the long running function call from the event
    if not event.long_running_tool_ids or not event.content or not event.content.parts:
        return
    for part in event.content.parts:
        if (
            part 
            and part.function_call 
            and event.long_running_tool_ids 
            and part.function_call.id in event.long_running_tool_ids
        ):
            return part.function_call

def get_function_response(event: Event, function_call_id: str) -> types.FunctionResponse:
    # Get the function response for the fuction call with specified id.
    if not event.content or not event.content.parts:
        return
    for part in event.content.parts:
        if (
            part 
            and part.function_response
            and part.function_response.id == function_call_id
        ):
            return part.function_response

print("\nRunning agent...")
events_async = runner.run_async(
    session_id=session.id, user_id='user', new_message=content
)


long_running_function_call, long_running_function_response, ticket_id = None, None, None
async for event in events_async:
    # Use helper to check for the specific auth request event
    if not long_running_function_call:
        long_running_function_call = get_long_running_function_call(event)
    else:
        long_running_function_response = get_function_response(event, long_running_function_call.id)
        if long_running_function_response:
            ticket_id = long_running_function_response.response['ticket_id']
    if event.content and event.content.parts:
        if text := ''.join(part.text or '' for part in event.content.parts):
            print(f'[{event.author}]: {text}')

    if long_running_function_response:
        # query the status of the correpsonding ticket via tciket_id
        # send back an intermediate / final response
        updated_response = long_running_function_response.model_copy(deep=True)
        updated_response.response = {'status': 'approved'}
        async for event in runner.run_async(
          session_id=session.id, user_id='user', new_message=types.Content(parts=[types.Part(function_response = updated_response)], role='user')
        ):
            if event.content and event.content.parts:
                if text := ''.join(part.text or '' for part in event.content.parts):
                    print(f'[{event.author}]: {text}')   
Example: File Processing Simulation

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from typing import Any
from google.adk.agents import Agent
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.tools import LongRunningFunctionTool
from google.adk.sessions import InMemorySessionService
from google.genai import types

# 1. Define the long running function
def ask_for_approval(
    purpose: str, amount: float
) -> dict[str, Any]:
    """Ask for approval for the reimbursement."""
    # create a ticket for the approval
    # Send a notification to the approver with the link of the ticket
    return {'status': 'pending', 'approver': 'Sean Zhou', 'purpose' : purpose, 'amount': amount, 'ticket-id': 'approval-ticket-1'}

def reimburse(purpose: str, amount: float) -> str:
    """Reimburse the amount of money to the employee."""
    # send the reimbrusement request to payment vendor
    return {'status': 'ok'}

# 2. Wrap the function with LongRunningFunctionTool
long_running_tool = LongRunningFunctionTool(func=ask_for_approval)

# 3. Use the tool in an Agent
file_processor_agent = Agent(
    # Use a model compatible with function calling
    model="gemini-2.0-flash",
    name='reimbursement_agent',
    instruction="""
      You are an agent whose job is to handle the reimbursement process for
      the employees. If the amount is less than $100, you will automatically
      approve the reimbursement.

      If the amount is greater than $100, you will
      ask for approval from the manager. If the manager approves, you will
      call reimburse() to reimburse the amount to the employee. If the manager
      rejects, you will inform the employee of the rejection.
    """,
    tools=[reimburse, long_running_tool]
)


APP_NAME = "human_in_the_loop"
USER_ID = "1234"
SESSION_ID = "session1234"

# Session and Runner
session_service = InMemorySessionService()
session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=file_processor_agent, app_name=APP_NAME, session_service=session_service)


# Agent Interaction
async def call_agent(query):

    def get_long_running_function_call(event: Event) -> types.FunctionCall:
        # Get the long running function call from the event
        if not event.long_running_tool_ids or not event.content or not event.content.parts:
            return
        for part in event.content.parts:
            if (
                part
                and part.function_call
                and event.long_running_tool_ids
                and part.function_call.id in event.long_running_tool_ids
            ):
                return part.function_call

    def get_function_response(event: Event, function_call_id: str) -> types.FunctionResponse:
        # Get the function response for the fuction call with specified id.
        if not event.content or not event.content.parts:
            return
        for part in event.content.parts:
            if (
                part
                and part.function_response
                and part.function_response.id == function_call_id
            ):
                return part.function_response

    content = types.Content(role='user', parts=[types.Part(text=query)])
    events = runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    print("\nRunning agent...")
    events_async = runner.run_async(
        session_id=session.id, user_id='user', new_message=content
    )


    long_running_function_call, long_running_function_response, ticket_id = None, None, None
    async for event in events_async:
        # Use helper to check for the specific auth request event
        if not long_running_function_call:
            long_running_function_call = get_long_running_function_call(event)
        else:
            long_running_function_response = get_function_response(event, long_running_function_call.id)
            if long_running_function_response:
                ticket_id = long_running_function_response.response['ticket_id']
        if event.content and event.content.parts:
            if text := ''.join(part.text or '' for part in event.content.parts):
                print(f'[{event.author}]: {text}')


    if long_running_function_response:
        # query the status of the correpsonding ticket via tciket_id
        # send back an intermediate / final response
        updated_response = long_running_function_response.model_copy(deep=True)
        updated_response.response = {'status': 'approved'}
        async for event in runner.run_async(
          session_id=session.id, user_id='user', new_message=types.Content(parts=[types.Part(function_response = updated_response)], role='user')
        ):
            if event.content and event.content.parts:
                if text := ''.join(part.text or '' for part in event.content.parts):
                    print(f'[{event.author}]: {text}')

# reimbursement that doesn't require approval
asyncio.run(call_agent("Please reimburse 50$ for meals"))
# reimbursement that requires approval
asyncio.run(call_agent("Please reimburse 200$ for meals"))
Key aspects of this example¶
process_large_file: This generator simulates a lengthy operation, yielding intermediate status/progress dictionaries.

LongRunningFunctionTool: Wraps the generator; the framework handles sending yielded updates and the final return value as sequential FunctionResponses.

Agent instruction: Directs the LLM to use the tool and understand the incoming FunctionResponse stream (progress vs. completion) for user updates.

Final return: The function returns the final result dictionary, which is sent in the concluding FunctionResponse to indicate completion.

3. Agent-as-a-Tool¶
This powerful feature allows you to leverage the capabilities of other agents within your system by calling them as tools. The Agent-as-a-Tool enables you to invoke another agent to perform a specific task, effectively delegating responsibility. This is conceptually similar to creating a Python function that calls another agent and uses the agent's response as the function's return value.

Key difference from sub-agents¶
It's important to distinguish an Agent-as-a-Tool from a Sub-Agent.

Agent-as-a-Tool: When Agent A calls Agent B as a tool (using Agent-as-a-Tool), Agent B's answer is passed back to Agent A, which then summarizes the answer and generates a response to the user. Agent A retains control and continues to handle future user input.

Sub-agent: When Agent A calls Agent B as a sub-agent, the responsibility of answering the user is completely transferred to Agent B. Agent A is effectively out of the loop. All subsequent user input will be answered by Agent B.

Usage¶
To use an agent as a tool, wrap the agent with the AgentTool class.


tools=[AgentTool(agent=agent_b)]
Customization¶
The AgentTool class provides the following attributes for customizing its behavior:

skip_summarization: bool: If set to True, the framework will bypass the LLM-based summarization of the tool agent's response. This can be useful when the tool's response is already well-formatted and requires no further processing.
Example

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

APP_NAME="summary_agent"
USER_ID="user1234"
SESSION_ID="1234"

summary_agent = Agent(
    model="gemini-2.0-flash",
    name="summary_agent",
    instruction="""You are an expert summarizer. Please read the following text and provide a concise summary.""",
    description="Agent to summarize text",
)

root_agent = Agent(
    model='gemini-2.0-flash',
    name='root_agent',
    instruction="""You are a helpful assistant. When the user provides a text, use the 'summarize' tool to generate a summary. Always forward the user's message exactly as received to the 'summarize' tool, without modifying or summarizing it yourself. Present the response from the tool to the user.""",
    tools=[AgentTool(agent=summary_agent)]
)

# Session and Runner
session_service = InMemorySessionService()
session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)


# Agent Interaction
def call_agent(query):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("Agent Response: ", final_response)


long_text = """Quantum computing represents a fundamentally different approach to computation, 
leveraging the bizarre principles of quantum mechanics to process information. Unlike classical computers 
that rely on bits representing either 0 or 1, quantum computers use qubits which can exist in a state of superposition - effectively 
being 0, 1, or a combination of both simultaneously. Furthermore, qubits can become entangled, 
meaning their fates are intertwined regardless of distance, allowing for complex correlations. This parallelism and 
interconnectedness grant quantum computers the potential to solve specific types of incredibly complex problems - such 
as drug discovery, materials science, complex system optimization, and breaking certain types of cryptography - far 
faster than even the most powerful classical supercomputers could ever achieve, although the technology is still largely in its developmental stages."""


call_agent(long_text)
How it works¶
When the main_agent receives the long text, its instruction tells it to use the 'summarize' tool for long texts.
The framework recognizes 'summarize' as an AgentTool that wraps the summary_agent.
Behind the scenes, the main_agent will call the summary_agent with the long text as input.
The summary_agent will process the text according to its instruction and generate a summary.
The response from the summary_agent is then passed back to the main_agent.
The main_agent can then take the summary and formulate its final response to the user (e.g., "Here's a summary of the text: ...")


Built-in tools¶
These built-in tools provide ready-to-use functionality such as Google Search or code executors that provide agents with common capabilities. For instance, an agent that needs to retrieve information from the web can directly use the google_search tool without any additional setup.

How to Use¶
Import: Import the desired tool from the agents.tools module.
Configure: Initialize the tool, providing required parameters if any.
Register: Add the initialized tool to the tools list of your Agent.
Once added to an agent, the agent can decide to use the tool based on the user prompt and its instructions. The framework handles the execution of the tool when the agent calls it. Important: check the Limitations section of this page.

Available Built-in tools¶
Google Search¶
The google_search tool allows the agent to perform web searches using Google Search. The google_search tool is only compatible with Gemini 2 models.

Additional requirements when using the google_search tool

When you use grounding with Google Search, and you receive Search suggestions in your response, you must display the Search suggestions in production and in your applications. For more information on grounding with Google Search, see Grounding with Google Search documentation for Google AI Studio or Vertex AI. The UI code (HTML) is returned in the Gemini response as renderedContent, and you will need to show the HTML in your app, in accordance with the policy.


from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types

APP_NAME="google_search_agent"
USER_ID="user1234"
SESSION_ID="1234"


root_agent = Agent(
    name="basic_search_agent",
    model="gemini-2.0-flash",
    description="Agent to answer questions using Google Search.",
    instruction="I can answer your questions by searching the internet. Just ask me anything!",
    # google_search is a pre-built tool which allows the agent to perform Google searches.
    tools=[google_search]
)

# Session and Runner
session_service = InMemorySessionService()
session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)


# Agent Interaction
def call_agent(query):
    """
    Helper function to call the agent with a query.
    """
    content = types.Content(role='user', parts=[types.Part(text=query)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("Agent Response: ", final_response)

call_agent("what's the latest ai news?")
Code Execution¶
The built_in_code_execution tool enables the agent to execute code, specifically when using Gemini 2 models. This allows the model to perform tasks like calculations, data manipulation, or running small scripts.


import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import built_in_code_execution
from google.genai import types

AGENT_NAME="calculator_agent"
APP_NAME="calculator"
USER_ID="user1234"
SESSION_ID="session_code_exec_async"
GEMINI_MODEL = "gemini-2.0-flash"

# Agent Definition
code_agent = LlmAgent(
    name=AGENT_NAME,
    model=GEMINI_MODEL,
    tools=[built_in_code_execution],
    instruction="""You are a calculator agent.
    When given a mathematical expression, write and execute Python code to calculate the result.
    Return only the final numerical result as plain text, without markdown or code blocks.
    """,
    description="Executes Python code to perform calculations.",
)

# Session and Runner
session_service = InMemorySessionService()
session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=code_agent, app_name=APP_NAME, session_service=session_service)

# Agent Interaction (Async)
async def call_agent_async(query):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    print(f"\n--- Running Query: {query} ---")
    final_response_text = "No final text response captured."
    try:
        # Use run_async
        async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
            print(f"Event ID: {event.id}, Author: {event.author}")

            # --- Check for specific parts FIRST ---
            has_specific_part = False
            if event.content and event.content.parts:
                for part in event.content.parts: # Iterate through all parts
                    if part.executable_code:
                        # Access the actual code string via .code
                        print(f"  Debug: Agent generated code:\n```python\n{part.executable_code.code}\n```")
                        has_specific_part = True
                    elif part.code_execution_result:
                        # Access outcome and output correctly
                        print(f"  Debug: Code Execution Result: {part.code_execution_result.outcome} - Output:\n{part.code_execution_result.output}")
                        has_specific_part = True
                    # Also print any text parts found in any event for debugging
                    elif part.text and not part.text.isspace():
                        print(f"  Text: '{part.text.strip()}'")
                        # Do not set has_specific_part=True here, as we want the final response logic below

            # --- Check for final response AFTER specific parts ---
            # Only consider it final if it doesn't have the specific code parts we just handled
            if not has_specific_part and event.is_final_response():
                if event.content and event.content.parts and event.content.parts[0].text:
                    final_response_text = event.content.parts[0].text.strip()
                    print(f"==> Final Agent Response: {final_response_text}")
                else:
                    print("==> Final Agent Response: [No text content in final event]")


    except Exception as e:
        print(f"ERROR during agent run: {e}")
    print("-" * 30)


# Main async function to run the examples
async def main():
    await call_agent_async("Calculate the value of (5 + 7) * 3")
    await call_agent_async("What is 10 factorial?")

# Execute the main async function
try:
    asyncio.run(main())
except RuntimeError as e:
    # Handle specific error when running asyncio.run in an already running loop (like Jupyter/Colab)
    if "cannot be called from a running event loop" in str(e):
        print("\nRunning in an existing event loop (like Colab/Jupyter).")
        print("Please run `await main()` in a notebook cell instead.")
        # If in an interactive environment like a notebook, you might need to run:
        # await main()
    else:
        raise e # Re-raise other runtime errors
Vertex AI Search¶
The vertex_ai_search_tool uses Google Cloud's Vertex AI Search, enabling the agent to search across your private, configured data stores (e.g., internal documents, company policies, knowledge bases). This built-in tool requires you to provide the specific data store ID during configuration.


import asyncio

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.tools import VertexAiSearchTool

# Replace with your actual Vertex AI Search Datastore ID
# Format: projects/<PROJECT_ID>/locations/<LOCATION>/collections/default_collection/dataStores/<DATASTORE_ID>
# e.g., "projects/12345/locations/us-central1/collections/default_collection/dataStores/my-datastore-123"
YOUR_DATASTORE_ID = "YOUR_DATASTORE_ID_HERE"

# Constants
APP_NAME_VSEARCH = "vertex_search_app"
USER_ID_VSEARCH = "user_vsearch_1"
SESSION_ID_VSEARCH = "session_vsearch_1"
AGENT_NAME_VSEARCH = "doc_qa_agent"
GEMINI_2_FLASH = "gemini-2.0-flash"

# Tool Instantiation
# You MUST provide your datastore ID here.
vertex_search_tool = VertexAiSearchTool(data_store_id=YOUR_DATASTORE_ID)

# Agent Definition
doc_qa_agent = LlmAgent(
    name=AGENT_NAME_VSEARCH,
    model=GEMINI_2_FLASH, # Requires Gemini model
    tools=[vertex_search_tool],
    instruction=f"""You are a helpful assistant that answers questions based on information found in the document store: {YOUR_DATASTORE_ID}.
    Use the search tool to find relevant information before answering.
    If the answer isn't in the documents, say that you couldn't find the information.
    """,
    description="Answers questions using a specific Vertex AI Search datastore.",
)

# Session and Runner Setup
session_service_vsearch = InMemorySessionService()
runner_vsearch = Runner(
    agent=doc_qa_agent, app_name=APP_NAME_VSEARCH, session_service=session_service_vsearch
)
session_vsearch = session_service_vsearch.create_session(
    app_name=APP_NAME_VSEARCH, user_id=USER_ID_VSEARCH, session_id=SESSION_ID_VSEARCH
)

# Agent Interaction Function
async def call_vsearch_agent_async(query):
    print("\n--- Running Vertex AI Search Agent ---")
    print(f"Query: {query}")
    if "YOUR_DATASTORE_ID_HERE" in YOUR_DATASTORE_ID:
        print("Skipping execution: Please replace YOUR_DATASTORE_ID_HERE with your actual datastore ID.")
        print("-" * 30)
        return

    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_response_text = "No response received."
    try:
        async for event in runner_vsearch.run_async(
            user_id=USER_ID_VSEARCH, session_id=SESSION_ID_VSEARCH, new_message=content
        ):
            # Like Google Search, results are often embedded in the model's response.
            if event.is_final_response() and event.content and event.content.parts:
                final_response_text = event.content.parts[0].text.strip()
                print(f"Agent Response: {final_response_text}")
                # You can inspect event.grounding_metadata for source citations
                if event.grounding_metadata:
                    print(f"  (Grounding metadata found with {len(event.grounding_metadata.grounding_attributions)} attributions)")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Ensure your datastore ID is correct and the service account has permissions.")
    print("-" * 30)

# --- Run Example ---
async def run_vsearch_example():
    # Replace with a question relevant to YOUR datastore content
    await call_vsearch_agent_async("Summarize the main points about the Q2 strategy document.")
    await call_vsearch_agent_async("What safety procedures are mentioned for lab X?")

# Execute the example
# await run_vsearch_example()

# Running locally due to potential colab asyncio issues with multiple awaits
try:
    asyncio.run(run_vsearch_example())
except RuntimeError as e:
    if "cannot be called from a running event loop" in str(e):
        print("Skipping execution in running event loop (like Colab/Jupyter). Run locally.")
    else:
        raise e
Use Built-in tools with other tools¶
The following code sample demonstrates how to use multiple built-in tools or how to use built-in tools with other tools by using multiple agents:


from google.adk.tools import agent_tool
from google.adk.agents import Agent
from google.adk.tools import google_search, built_in_code_execution

search_agent = Agent(
    model='gemini-2.0-flash',
    name='SearchAgent',
    instruction="""
    You're a specialist in Google Search
    """,
    tools=[google_search],
)
coding_agent = Agent(
    model='gemini-2.0-flash',
    name='CodeAgent',
    instruction="""
    You're a specialist in Code Execution
    """,
    tools=[built_in_code_execution],
)
root_agent = Agent(
    name="RootAgent",
    model="gemini-2.0-flash",
    description="Root Agent",
    tools=[agent_tool.AgentTool(agent=search_agent), agent_tool.AgentTool(agent=coding_agent)],
)
Limitations¶
Warning

Currently, for each root agent or single agent, only one built-in tool is supported. No other tools of any type can be used in the same agent.

For example, the following approach that uses a built-in tool along with other tools within a single agent is not currently supported:


root_agent = Agent(
    name="RootAgent",
    model="gemini-2.0-flash",
    description="Root Agent",
    tools=[built_in_code_execution, custom_function], # <-- not supported
)
Warning

Built-in tools cannot be used within a sub-agent.

For example, the following approach that uses built-in tools within sub-agents is not currently supported:


search_agent = Agent(
    model='gemini-2.0-flash',
    name='SearchAgent',
    instruction="""
    You're a specialist in Google Search
    """,
    tools=[google_search],
)
coding_agent = Agent(
    model='gemini-2.0-flash',
    name='CodeAgent',
    instruction="""
    You're a specialist in Code Execution
    """,
    tools=[built_in_code_execution],
)
root_agent = Agent(
    name="RootAgent",
    model="gemini-2.0-flash",
    description="Root Agent",
    sub_agents=[
        search_agent,
        coding_agent
    ],
)

Runtime¶
What is runtime?¶
The ADK Runtime is the underlying engine that powers your agent application during user interactions. It's the system that takes your defined agents, tools, and callbacks and orchestrates their execution in response to user input, managing the flow of information, state changes, and interactions with external services like LLMs or storage.

Think of the Runtime as the "engine" of your agentic application. You define the parts (agents, tools), and the Runtime handles how they connect and run together to fulfill a user's request.

Core Idea: The Event Loop¶
At its heart, the ADK Runtime operates on an Event Loop. This loop facilitates a back-and-forth communication between the Runner component and your defined "Execution Logic" (which includes your Agents, the LLM calls they make, Callbacks, and Tools).

intro_components.png

In simple terms:

The Runner receives a user query and asks the main Agent to start processing.
The Agent (and its associated logic) runs until it has something to report (like a response, a request to use a tool, or a state change) – it then yields an Event.
The Runner receives this Event, processes any associated actions (like saving state changes via Services), and forwards the event onwards (e.g., to the user interface).
Only after the Runner has processed the event does the Agent's logic resume from where it paused, now potentially seeing the effects of the changes committed by the Runner.
This cycle repeats until the agent has no more events to yield for the current user query.
This event-driven loop is the fundamental pattern governing how ADK executes your agent code.

The Heartbeat: The Event Loop - Inner workings¶
The Event Loop is the core operational pattern defining the interaction between the Runner and your custom code (Agents, Tools, Callbacks, collectively referred to as "Execution Logic" or "Logic Components" in the design document). It establishes a clear division of responsibilities:

Runner's Role (Orchestrator)¶
The Runner acts as the central coordinator for a single user invocation. Its responsibilities in the loop are:

Initiation: Receives the end user's query (new_message) and typically appends it to the session history via the SessionService.
Kick-off: Starts the event generation process by calling the main agent's execution method (e.g., agent_to_run.run_async(...)).
Receive & Process: Waits for the agent logic to yield an Event. Upon receiving an event, the Runner promptly processes it. This involves:
Using configured Services (SessionService, ArtifactService, MemoryService) to commit changes indicated in event.actions (like state_delta, artifact_delta).
Performing other internal bookkeeping.
Yield Upstream: Forwards the processed event onwards (e.g., to the calling application or UI for rendering).
Iterate: Signals the agent logic that processing is complete for the yielded event, allowing it to resume and generate the next event.
Conceptual Runner Loop:


# Simplified view of Runner's main loop logic
def run(new_query, ...) -> Generator[Event]:
    # 1. Append new_query to session event history (via SessionService)
    session_service.append_event(session, Event(author='user', content=new_query))

    # 2. Kick off event loop by calling the agent
    agent_event_generator = agent_to_run.run_async(context)

    async for event in agent_event_generator:
        # 3. Process the generated event and commit changes
        session_service.append_event(session, event) # Commits state/artifact deltas etc.
        # memory_service.update_memory(...) # If applicable
        # artifact_service might have already been called via context during agent run

        # 4. Yield event for upstream processing (e.g., UI rendering)
        yield event
        # Runner implicitly signals agent generator can continue after yielding
Execution Logic's Role (Agent, Tool, Callback)¶
Your code within agents, tools, and callbacks is responsible for the actual computation and decision-making. Its interaction with the loop involves:

Execute: Runs its logic based on the current InvocationContext, including the session state as it was when execution resumed.
Yield: When the logic needs to communicate (send a message, call a tool, report a state change), it constructs an Event containing the relevant content and actions, and then yields this event back to the Runner.
Pause: Crucially, execution of the agent logic pauses immediately after the yield statement. It waits for the Runner to complete step 3 (processing and committing).
Resume: Only after the Runner has processed the yielded event does the agent logic resume execution from the statement immediately following the yield.
See Updated State: Upon resumption, the agent logic can now reliably access the session state (ctx.session.state) reflecting the changes that were committed by the Runner from the previously yielded event.
Conceptual Execution Logic:


# Simplified view of logic inside Agent.run_async, callbacks, or tools

# ... previous code runs based on current state ...

# 1. Determine a change or output is needed, construct the event
# Example: Updating state
update_data = {'field_1': 'value_2'}
event_with_state_change = Event(
    author=self.name,
    actions=EventActions(state_delta=update_data),
    content=types.Content(parts=[types.Part(text="State updated.")])
    # ... other event fields ...
)

# 2. Yield the event to the Runner for processing & commit
yield event_with_state_change
# <<<<<<<<<<<< EXECUTION PAUSES HERE >>>>>>>>>>>>

# <<<<<<<<<<<< RUNNER PROCESSES & COMMITS THE EVENT >>>>>>>>>>>>

# 3. Resume execution ONLY after Runner is done processing the above event.
# Now, the state committed by the Runner is reliably reflected.
# Subsequent code can safely assume the change from the yielded event happened.
val = ctx.session.state['field_1']
# here `val` is guaranteed to be "value_2" (assuming Runner committed successfully)
print(f"Resumed execution. Value of field_1 is now: {val}")

# ... subsequent code continues ...
# Maybe yield another event later...
This cooperative yield/pause/resume cycle between the Runner and your Execution Logic, mediated by Event objects, forms the core of the ADK Runtime.

Key components of the Runtime¶
Several components work together within the ADK Runtime to execute an agent invocation. Understanding their roles clarifies how the event loop functions:

Runner¶
Role: The main entry point and orchestrator for a single user query (run_async).
Function: Manages the overall Event Loop, receives events yielded by the Execution Logic, coordinates with Services to process and commit event actions (state/artifact changes), and forwards processed events upstream (e.g., to the UI). It essentially drives the conversation turn by turn based on yielded events. (Defined in google.adk.runners.runner.py).
Execution Logic Components¶
Role: The parts containing your custom code and the core agent capabilities.
Components:
Agent (BaseAgent, LlmAgent, etc.): Your primary logic units that process information and decide on actions. They implement the _run_async_impl method which yields events.
Tools (BaseTool, FunctionTool, AgentTool, etc.): External functions or capabilities used by agents (often LlmAgent) to interact with the outside world or perform specific tasks. They execute and return results, which are then wrapped in events.
Callbacks (Functions): User-defined functions attached to agents (e.g., before_agent_callback, after_model_callback) that hook into specific points in the execution flow, potentially modifying behavior or state, whose effects are captured in events.
Function: Perform the actual thinking, calculation, or external interaction. They communicate their results or needs by yielding Event objects and pausing until the Runner processes them.
Event¶
Role: The message passed back and forth between the Runner and the Execution Logic.
Function: Represents an atomic occurrence (user input, agent text, tool call/result, state change request, control signal). It carries both the content of the occurrence and the intended side effects (actions like state_delta). (Defined in google.adk.events.event.py).
Services¶
Role: Backend components responsible for managing persistent or shared resources. Used primarily by the Runner during event processing.
Components:
SessionService (BaseSessionService, InMemorySessionService, etc.): Manages Session objects, including saving/loading them, applying state_delta to the session state, and appending events to the event history.
ArtifactService (BaseArtifactService, InMemoryArtifactService, GcsArtifactService, etc.): Manages the storage and retrieval of binary artifact data. Although save_artifact is called via context during execution logic, the artifact_delta in the event confirms the action for the Runner/SessionService.
MemoryService (BaseMemoryService, etc.): (Optional) Manages long-term semantic memory across sessions for a user.
Function: Provide the persistence layer. The Runner interacts with them to ensure changes signaled by event.actions are reliably stored before the Execution Logic resumes.
Session¶
Role: A data container holding the state and history for one specific conversation between a user and the application.
Function: Stores the current state dictionary, the list of all past events (event history), and references to associated artifacts. It's the primary record of the interaction, managed by the SessionService. (Defined in google.adk.sessions.session.py).
Invocation¶
Role: A conceptual term representing everything that happens in response to a single user query, from the moment the Runner receives it until the agent logic finishes yielding events for that query.
Function: An invocation might involve multiple agent runs (if using agent transfer or AgentTool), multiple LLM calls, tool executions, and callback executions, all tied together by a single invocation_id within the InvocationContext.
These players interact continuously through the Event Loop to process a user's request.

How It Works: A Simplified Invocation¶
Let's trace a simplified flow for a typical user query that involves an LLM agent calling a tool:

intro_components.png

Step-by-Step Breakdown¶
User Input: The User sends a query (e.g., "What's the capital of France?").
Runner Starts: Runner.run_async begins. It interacts with the SessionService to load the relevant Session and adds the user query as the first Event to the session history. An InvocationContext (ctx) is prepared.
Agent Execution: The Runner calls agent.run_async(ctx) on the designated root agent (e.g., an LlmAgent).
LLM Call (Example): The Agent_Llm determines it needs information, perhaps by calling a tool. It prepares a request for the LLM. Let's assume the LLM decides to call MyTool.
Yield FunctionCall Event: The Agent_Llm receives the FunctionCall response from the LLM, wraps it in an Event(author='Agent_Llm', content=Content(parts=[Part(function_call=...)])), and yields this event.
Agent Pauses: The Agent_Llm's execution pauses immediately after the yield.
Runner Processes: The Runner receives the FunctionCall event. It passes it to the SessionService to record it in the history. The Runner then yields the event upstream to the User (or application).
Agent Resumes: The Runner signals that the event is processed, and Agent_Llm resumes execution.
Tool Execution: The Agent_Llm's internal flow now proceeds to execute the requested MyTool. It calls tool.run_async(...).
Tool Returns Result: MyTool executes and returns its result (e.g., {'result': 'Paris'}).
Yield FunctionResponse Event: The agent (Agent_Llm) wraps the tool result into an Event containing a FunctionResponse part (e.g., Event(author='Agent_Llm', content=Content(role='user', parts=[Part(function_response=...)]))). This event might also contain actions if the tool modified state (state_delta) or saved artifacts (artifact_delta). The agent yields this event.
Agent Pauses: Agent_Llm pauses again.
Runner Processes: Runner receives the FunctionResponse event. It passes it to SessionService which applies any state_delta/artifact_delta and adds the event to history. Runner yields the event upstream.
Agent Resumes: Agent_Llm resumes, now knowing the tool result and any state changes are committed.
Final LLM Call (Example): Agent_Llm sends the tool result back to the LLM to generate a natural language response.
Yield Final Text Event: Agent_Llm receives the final text from the LLM, wraps it in an Event(author='Agent_Llm', content=Content(parts=[Part(text=...)])), and yields it.
Agent Pauses: Agent_Llm pauses.
Runner Processes: Runner receives the final text event, passes it to SessionService for history, and yields it upstream to the User. This is likely marked as the is_final_response().
Agent Resumes & Finishes: Agent_Llm resumes. Having completed its task for this invocation, its run_async generator finishes.
Runner Completes: The Runner sees the agent's generator is exhausted and finishes its loop for this invocation.
This yield/pause/process/resume cycle ensures that state changes are consistently applied and that the execution logic always operates on the most recently committed state after yielding an event.

Important Runtime Behaviors¶
Understanding a few key aspects of how the ADK Runtime handles state, streaming, and asynchronous operations is crucial for building predictable and efficient agents.

State Updates & Commitment Timing¶
The Rule: When your code (in an agent, tool, or callback) modifies the session state (e.g., context.state['my_key'] = 'new_value'), this change is initially recorded locally within the current InvocationContext. The change is only guaranteed to be persisted (saved by the SessionService) after the Event carrying the corresponding state_delta in its actions has been yield-ed by your code and subsequently processed by the Runner.

Implication: Code that runs after resuming from a yield can reliably assume that the state changes signaled in the yielded event have been committed.


# Inside agent logic (conceptual)

# 1. Modify state
ctx.session.state['status'] = 'processing'
event1 = Event(..., actions=EventActions(state_delta={'status': 'processing'}))

# 2. Yield event with the delta
yield event1
# --- PAUSE --- Runner processes event1, SessionService commits 'status' = 'processing' ---

# 3. Resume execution
# Now it's safe to rely on the committed state
current_status = ctx.session.state['status'] # Guaranteed to be 'processing'
print(f"Status after resuming: {current_status}")
"Dirty Reads" of Session State¶
Definition: While commitment happens after the yield, code running later within the same invocation, but before the state-changing event is actually yielded and processed, can often see the local, uncommitted changes. This is sometimes called a "dirty read".
Example:

# Code in before_agent_callback
callback_context.state['field_1'] = 'value_1'
# State is locally set to 'value_1', but not yet committed by Runner

# ... agent runs ...

# Code in a tool called later *within the same invocation*
# Readable (dirty read), but 'value_1' isn't guaranteed persistent yet.
val = tool_context.state['field_1'] # 'val' will likely be 'value_1' here
print(f"Dirty read value in tool: {val}")

# Assume the event carrying the state_delta={'field_1': 'value_1'}
# is yielded *after* this tool runs and is processed by the Runner.
Implications:
Benefit: Allows different parts of your logic within a single complex step (e.g., multiple callbacks or tool calls before the next LLM turn) to coordinate using state without waiting for a full yield/commit cycle.
Caveat: Relying heavily on dirty reads for critical logic can be risky. If the invocation fails before the event carrying the state_delta is yielded and processed by the Runner, the uncommitted state change will be lost. For critical state transitions, ensure they are associated with an event that gets successfully processed.
Streaming vs. Non-Streaming Output (partial=True)¶
This primarily relates to how responses from the LLM are handled, especially when using streaming generation APIs.

Streaming: The LLM generates its response token-by-token or in small chunks.
The framework (often within BaseLlmFlow) yields multiple Event objects for a single conceptual response. Most of these events will have partial=True.
The Runner, upon receiving an event with partial=True, typically forwards it immediately upstream (for UI display) but skips processing its actions (like state_delta).
Eventually, the framework yields a final event for that response, marked as non-partial (partial=False or implicitly via turn_complete=True).
The Runner fully processes only this final event, committing any associated state_delta or artifact_delta.
Non-Streaming: The LLM generates the entire response at once. The framework yields a single event marked as non-partial, which the Runner processes fully.
Why it Matters: Ensures that state changes are applied atomically and only once based on the complete response from the LLM, while still allowing the UI to display text progressively as it's generated.
Async is Primary (run_async)¶
Core Design: The ADK Runtime is fundamentally built on Python's asyncio library to handle concurrent operations (like waiting for LLM responses or tool executions) efficiently without blocking.
Main Entry Point: Runner.run_async is the primary method for executing agent invocations. All core runnable components (Agents, specific flows) use async def methods internally.
Synchronous Convenience (run): A synchronous Runner.run method exists mainly for convenience (e.g., in simple scripts or testing environments). However, internally, Runner.run typically just calls Runner.run_async and manages the async event loop execution for you.
Developer Experience: You should generally design your application logic (e.g., web servers using ADK) using asyncio.
Sync Callbacks/Tools: The framework aims to handle both async def and regular def functions provided as tools or callbacks seamlessly. Long-running synchronous tools or callbacks, especially those performing blocking I/O, can potentially block the main asyncio event loop. The framework might use mechanisms like asyncio.to_thread to mitigate this by running such blocking synchronous code in a separate thread pool, preventing it from stalling other asynchronous tasks. CPU-bound synchronous code, however, will still block the thread it runs on.
Understanding these behaviors helps you write more robust ADK applications and debug issues related to state consistency, streaming updates, and asynchronous execution.

Runtime Configuration¶
RunConfig defines runtime behavior and options for agents in the ADK. It controls speech and streaming settings, function calling, artifact saving, and limits on LLM calls.

When constructing an agent run, you can pass a RunConfig to customize how the agent interacts with models, handles audio, and streams responses. By default, no streaming is enabled and inputs aren’t retained as artifacts. Use RunConfig to override these defaults.

Class Definition¶
The RunConfig class is a Pydantic model that enforces strict validation of configuration parameters.


class RunConfig(BaseModel):
    """Configs for runtime behavior of agents."""

    model_config = ConfigDict(
        extra='forbid',
    )

    speech_config: Optional[types.SpeechConfig] = None
    response_modalities: Optional[list[str]] = None
    save_input_blobs_as_artifacts: bool = False
    support_cfc: bool = False
    streaming_mode: StreamingMode = StreamingMode.NONE
    output_audio_transcription: Optional[types.AudioTranscriptionConfig] = None
    max_llm_calls: int = 500
Runtime Parameters¶
Parameter	Type	Default	Description
speech_config	Optional[types.SpeechConfig]	None	Configures speech synthesis (voice, language) via nested types.SpeechConfig.
response_modalities	Optional[list[str]]	None	List of desired output modalities (e.g., ["TEXT", "AUDIO"]). Default is None.
save_input_blobs_as_artifacts	bool	False	If True, saves input blobs (e.g., uploaded files) as run artifacts for debugging/auditing.
support_cfc	bool	False	Enables Compositional Function Calling. Requires streaming_mode=SSE and uses the LIVE API. Experimental.
streaming_mode	StreamingMode	StreamingMode.NONE	Sets the streaming behavior: NONE (default), SSE (server-sent events), or BIDI (bidirectional).
output_audio_transcription	Optional[types.AudioTranscriptionConfig]	None	Configures transcription of generated audio output via types.AudioTranscriptionConfig.
max_llm_calls	int	500	Limits total LLM calls per run. 0 or negative means unlimited (warned); sys.maxsize raises ValueError.
speech_config¶
Speech configuration settings for live agents with audio capabilities. The SpeechConfig class has the following structure:


class SpeechConfig(_common.BaseModel):
    """The speech generation configuration."""

    voice_config: Optional[VoiceConfig] = Field(
        default=None,
        description="""The configuration for the speaker to use.""",
    )
    language_code: Optional[str] = Field(
        default=None,
        description="""Language code (ISO 639. e.g. en-US) for the speech synthesization.
        Only available for Live API.""",
    )
The voice_config parameter uses the VoiceConfig class:


class VoiceConfig(_common.BaseModel):
    """The configuration for the voice to use."""

    prebuilt_voice_config: Optional[PrebuiltVoiceConfig] = Field(
        default=None,
        description="""The configuration for the speaker to use.""",
    )
And PrebuiltVoiceConfig has the following structure:


class PrebuiltVoiceConfig(_common.BaseModel):
    """The configuration for the prebuilt speaker to use."""

    voice_name: Optional[str] = Field(
        default=None,
        description="""The name of the prebuilt voice to use.""",
    )
These nested configuration classes allow you to specify:

voice_config: The name of the prebuilt voice to use (in the PrebuiltVoiceConfig)
language_code: ISO 639 language code (e.g., "en-US") for speech synthesis
When implementing voice-enabled agents, configure these parameters to control how your agent sounds when speaking.

response_modalities¶
Defines the output modalities for the agent. If not set, defaults to AUDIO. Response modalities determine how the agent communicates with users through various channels (e.g., text, audio).

save_input_blobs_as_artifacts¶
When enabled, input blobs will be saved as artifacts during agent execution. This is useful for debugging and audit purposes, allowing developers to review the exact data received by agents.

support_cfc¶
Enables Compositional Function Calling (CFC) support. Only applicable when using StreamingMode.SSE. When enabled, the LIVE API will be invoked as only it supports CFC functionality.

Warning

The support_cfc feature is experimental and its API or behavior might change in future releases.

streaming_mode¶
Configures the streaming behavior of the agent. Possible values:

StreamingMode.NONE: No streaming; responses delivered as complete units
StreamingMode.SSE: Server-Sent Events streaming; one-way streaming from server to client
StreamingMode.BIDI: Bidirectional streaming; simultaneous communication in both directions
Streaming modes affect both performance and user experience. SSE streaming lets users see partial responses as they're generated, while BIDI streaming enables real-time interactive experiences.

output_audio_transcription¶
Configuration for transcribing audio outputs from live agents with audio response capability. This enables automatic transcription of audio responses for accessibility, record-keeping, and multi-modal applications.

max_llm_calls¶
Sets a limit on the total number of LLM calls for a given agent run.

Values greater than 0 and less than sys.maxsize: Enforces a bound on LLM calls
Values less than or equal to 0: Allows unbounded LLM calls (not recommended for production)
This parameter prevents excessive API usage and potential runaway processes. Since LLM calls often incur costs and consume resources, setting appropriate limits is crucial.

Validation Rules¶
As a Pydantic model, RunConfig automatically validates parameter types. In addition, it includes specific validation logic for the max_llm_calls parameter:

If set to sys.maxsize, a ValueError is raised to prevent integer overflow issues
If less than or equal to 0, a warning is logged about potential unlimited LLM calls
Examples¶
Basic runtime configuration¶

from google.genai.adk import RunConfig, StreamingMode

config = RunConfig(
    streaming_mode=StreamingMode.NONE,
    max_llm_calls=100
)
This configuration creates a non-streaming agent with a limit of 100 LLM calls, suitable for simple task-oriented agents where complete responses are preferable.

Enabling streaming¶

from google.genai.adk import RunConfig, StreamingMode

config = RunConfig(
    streaming_mode=StreamingMode.SSE,
    max_llm_calls=200
)
Using SSE streaming allows users to see responses as they're generated, providing a more responsive feel for chatbots and assistants.
Enabling speech support¶

from google.genai.adk import RunConfig, StreamingMode
from google.genai import types

config = RunConfig(
    speech_config=types.SpeechConfig(
        language_code="en-US",
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name="Kore"
            )
        ),
    ),
    response_modalities=["AUDIO", "TEXT"],
    save_input_blobs_as_artifacts=True,
    support_cfc=True,
    streaming_mode=StreamingMode.SSE,
    max_llm_calls=1000,
)
This comprehensive example configures an agent with:

Speech capabilities using the "Kore" voice (US English)
Both audio and text output modalities
Artifact saving for input blobs (useful for debugging)
Experimental CFC support enabled
SSE streaming for responsive interaction
A limit of 1000 LLM calls
Enabling Experimental CFC Support¶

from google.genai.adk import RunConfig, StreamingMode

config = RunConfig(
    streaming_mode=StreamingMode.SSE,
    support_cfc=True,
    max_llm_calls=150
)
Enabling Compositional Function Calling creates an agent that can dynamically execute functions based on model outputs, powerful for applications requiring complex workflows.

Callbacks: Observe, Customize, and Control Agent Behavior¶
Introduction: What are Callbacks and Why Use Them?¶
Callbacks are a cornerstone feature of ADK, providing a powerful mechanism to hook into an agent's execution process. They allow you to observe, customize, and even control the agent's behavior at specific, predefined points without modifying the core ADK framework code.

What are they? In essence, callbacks are standard Python functions that you define. You then associate these functions with an agent when you create it. The ADK framework automatically calls your functions at key stages, letting you observe or intervene. Think of it like checkpoints during the agent's process:

Before the agent starts its main work on a request, and after it finishes: When you ask an agent to do something (e.g., answer a question), it runs its internal logic to figure out the response.
The before_agent callback executes right before this main work begins for that specific request.
The after_agent callback executes right after the agent has finished all its steps for that request and has prepared the final result, but just before the result is returned.
This "main work" encompasses the agent's entire process for handling that single request. This might involve deciding to call an LLM, actually calling the LLM, deciding to use a tool, using the tool, processing the results, and finally putting together the answer. These callbacks essentially wrap the whole sequence from receiving the input to producing the final output for that one interaction.
Before sending a request to, or after receiving a response from, the Large Language Model (LLM): These callbacks (before_model, after_model) allow you to inspect or modify the data going to and coming from the LLM specifically.
Before executing a tool (like a Python function or another agent) or after it finishes: Similarly, before_tool and after_tool callbacks give you control points specifically around the execution of tools invoked by the agent.
intro_components.png

Why use them? Callbacks unlock significant flexibility and enable advanced agent capabilities:

Observe & Debug: Log detailed information at critical steps for monitoring and troubleshooting.
Customize & Control: Modify data flowing through the agent (like LLM requests or tool results) or even bypass certain steps entirely based on your logic.
Implement Guardrails: Enforce safety rules, validate inputs/outputs, or prevent disallowed operations.
Manage State: Read or dynamically update the agent's session state during execution.
Integrate & Enhance: Trigger external actions (API calls, notifications) or add features like caching.
How are they added? You register callbacks by passing your defined Python functions as arguments to the agent's constructor (__init__) when you create an instance of Agent or LlmAgent.


from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from typing import Optional

# --- Define your callback function ---
def my_before_model_logic(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    print(f"Callback running before model call for agent: {callback_context.agent_name}")
    # ... your custom logic here ...
    return None # Allow the model call to proceed

# --- Register it during Agent creation ---
my_agent = LlmAgent(
    name="MyCallbackAgent",
    model="gemini-2.0-flash", # Or your desired model
    instruction="Be helpful.",
    # Other agent parameters...
    before_model_callback=my_before_model_logic # Pass the function here
)
The Callback Mechanism: Interception and Control¶
When the ADK framework encounters a point where a callback can run (e.g., just before calling the LLM), it checks if you provided a corresponding callback function for that agent. If you did, the framework executes your function.

Context is Key: Your callback function isn't called in isolation. The framework provides special context objects (CallbackContext or ToolContext) as arguments. These objects contain vital information about the current state of the agent's execution, including the invocation details, session state, and potentially references to services like artifacts or memory. You use these context objects to understand the situation and interact with the framework. (See the dedicated "Context Objects" section for full details).

Controlling the Flow (The Core Mechanism): The most powerful aspect of callbacks lies in how their return value influences the agent's subsequent actions. This is how you intercept and control the execution flow:

return None (Allow Default Behavior):

This is the standard way to signal that your callback has finished its work (e.g., logging, inspection, minor modifications to mutable input arguments like llm_request) and that the ADK agent should proceed with its normal operation.
For before_* callbacks (before_agent, before_model, before_tool), returning None means the next step in the sequence (running the agent logic, calling the LLM, executing the tool) will occur.
For after_* callbacks (after_agent, after_model, after_tool), returning None means the result just produced by the preceding step (the agent's output, the LLM's response, the tool's result) will be used as is.
return <Specific Object> (Override Default Behavior):

Returning a specific type of object (instead of None) is how you override the ADK agent's default behavior. The framework will use the object you return and skip the step that would normally follow or replace the result that was just generated.
before_agent_callback → types.Content: Skips the agent's main execution logic (_run_async_impl / _run_live_impl). The returned Content object is immediately treated as the agent's final output for this turn. Useful for handling simple requests directly or enforcing access control.
before_model_callback → LlmResponse: Skips the call to the external Large Language Model. The returned LlmResponse object is processed as if it were the actual response from the LLM. Ideal for implementing input guardrails, prompt validation, or serving cached responses.
before_tool_callback → dict: Skips the execution of the actual tool function (or sub-agent). The returned dict is used as the result of the tool call, which is then typically passed back to the LLM. Perfect for validating tool arguments, applying policy restrictions, or returning mocked/cached tool results.
after_agent_callback → types.Content: Replaces the Content that the agent's run logic just produced.
after_model_callback → LlmResponse: Replaces the LlmResponse received from the LLM. Useful for sanitizing outputs, adding standard disclaimers, or modifying the LLM's response structure.
after_tool_callback → dict: Replaces the dict result returned by the tool. Allows for post-processing or standardization of tool outputs before they are sent back to the LLM.
Conceptual Code Example (Guardrail):

This example demonstrates the common pattern for a guardrail using before_model_callback.


from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.adk.runners import Runner
from typing import Optional
from google.genai import types 
from google.adk.sessions import InMemorySessionService

GEMINI_2_FLASH="gemini-2.0-flash"

# --- Define the Callback Function ---
def simple_before_model_modifier(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Inspects/modifies the LLM request or skips the call."""
    agent_name = callback_context.agent_name
    print(f"[Callback] Before model call for agent: {agent_name}")

    # Inspect the last user message in the request contents
    last_user_message = ""
    if llm_request.contents and llm_request.contents[-1].role == 'user':
         if llm_request.contents[-1].parts:
            last_user_message = llm_request.contents[-1].parts[0].text
    print(f"[Callback] Inspecting last user message: '{last_user_message}'")

    # --- Modification Example ---
    # Add a prefix to the system instruction
    original_instruction = llm_request.config.system_instruction or types.Content(role="system", parts=[])
    prefix = "[Modified by Callback] "
    # Ensure system_instruction is Content and parts list exists
    if not isinstance(original_instruction, types.Content):
         # Handle case where it might be a string (though config expects Content)
         original_instruction = types.Content(role="system", parts=[types.Part(text=str(original_instruction))])
    if not original_instruction.parts:
        original_instruction.parts.append(types.Part(text="")) # Add an empty part if none exist

    # Modify the text of the first part
    modified_text = prefix + (original_instruction.parts[0].text or "")
    original_instruction.parts[0].text = modified_text
    llm_request.config.system_instruction = original_instruction
    print(f"[Callback] Modified system instruction to: '{modified_text}'")

    # --- Skip Example ---
    # Check if the last user message contains "BLOCK"
    if "BLOCK" in last_user_message.upper():
        print("[Callback] 'BLOCK' keyword found. Skipping LLM call.")
        # Return an LlmResponse to skip the actual LLM call
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="LLM call was blocked by before_model_callback.")],
            )
        )
    else:
        print("[Callback] Proceeding with LLM call.")
        # Return None to allow the (modified) request to go to the LLM
        return None


# Create LlmAgent and Assign Callback
my_llm_agent = LlmAgent(
        name="ModelCallbackAgent",
        model=GEMINI_2_FLASH,
        instruction="You are a helpful assistant.", # Base instruction
        description="An LLM agent demonstrating before_model_callback",
        before_model_callback=simple_before_model_modifier # Assign the function here
)

APP_NAME = "guardrail_app"
USER_ID = "user_1"
SESSION_ID = "session_001"

# Session and Runner
session_service = InMemorySessionService()
session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=my_llm_agent, app_name=APP_NAME, session_service=session_service)


# Agent Interaction
def call_agent(query):
  content = types.Content(role='user', parts=[types.Part(text=query)])
  events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

  for event in events:
      if event.is_final_response():
          final_response = event.content.parts[0].text
          print("Agent Response: ", final_response)

call_agent("callback example")
By understanding this mechanism of returning None versus returning specific objects, you can precisely control the agent's execution path, making callbacks an essential tool for building sophisticated and reliable agents with ADK.

LLM Agent¶
The LlmAgent (often aliased simply as Agent) is a core component in ADK, acting as the "thinking" part of your application. It leverages the power of a Large Language Model (LLM) for reasoning, understanding natural language, making decisions, generating responses, and interacting with tools.

Unlike deterministic Workflow Agents that follow predefined execution paths, LlmAgent behavior is non-deterministic. It uses the LLM to interpret instructions and context, deciding dynamically how to proceed, which tools to use (if any), or whether to transfer control to another agent.

Building an effective LlmAgent involves defining its identity, clearly guiding its behavior through instructions, and equipping it with the necessary tools and capabilities.

Defining the Agent's Identity and Purpose¶
First, you need to establish what the agent is and what it's for.

name (Required): Every agent needs a unique string identifier. This name is crucial for internal operations, especially in multi-agent systems where agents need to refer to or delegate tasks to each other. Choose a descriptive name that reflects the agent's function (e.g., customer_support_router, billing_inquiry_agent). Avoid reserved names like user.

description (Optional, Recommended for Multi-Agent): Provide a concise summary of the agent's capabilities. This description is primarily used by other LLM agents to determine if they should route a task to this agent. Make it specific enough to differentiate it from peers (e.g., "Handles inquiries about current billing statements," not just "Billing agent").

model (Required): Specify the underlying LLM that will power this agent's reasoning. This is a string identifier like "gemini-2.0-flash". The choice of model impacts the agent's capabilities, cost, and performance. See the Models page for available options and considerations.


# Example: Defining the basic identity
capital_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="capital_agent",
    description="Answers user questions about the capital city of a given country."
    # instruction and tools will be added next
)
Guiding the Agent: Instructions (instruction)¶
The instruction parameter is arguably the most critical for shaping an LlmAgent's behavior. It's a string (or a function returning a string) that tells the agent:

Its core task or goal.
Its personality or persona (e.g., "You are a helpful assistant," "You are a witty pirate").
Constraints on its behavior (e.g., "Only answer questions about X," "Never reveal Y").
How and when to use its tools. You should explain the purpose of each tool and the circumstances under which it should be called, supplementing any descriptions within the tool itself.
The desired format for its output (e.g., "Respond in JSON," "Provide a bulleted list").
Tips for Effective Instructions:

Be Clear and Specific: Avoid ambiguity. Clearly state the desired actions and outcomes.
Use Markdown: Improve readability for complex instructions using headings, lists, etc.
Provide Examples (Few-Shot): For complex tasks or specific output formats, include examples directly in the instruction.
Guide Tool Use: Don't just list tools; explain when and why the agent should use them.
State:

The instruction is a string template, you can use the {var} syntax to insert dynamic values into the instruction.
{var} is used to insert the value of the state variable named var.
{artifact.var} is used to insert the text content of the artifact named var.
If the state variable or artifact does not exist, the agent will raise an error. If you want to ignore the error, you can append a ? to the variable name as in {var?}.

# Example: Adding instructions
capital_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="capital_agent",
    description="Answers user questions about the capital city of a given country.",
    instruction="""You are an agent that provides the capital city of a country.
When a user asks for the capital of a country:
1. Identify the country name from the user's query.
2. Use the `get_capital_city` tool to find the capital.
3. Respond clearly to the user, stating the capital city.
Example Query: "What's the capital of France?"
Example Response: "The capital of France is Paris."
""",
    # tools will be added next
)
(Note: For instructions that apply to all agents in a system, consider using global_instruction on the root agent, detailed further in the Multi-Agents section.)

Equipping the Agent: Tools (tools)¶
Tools give your LlmAgent capabilities beyond the LLM's built-in knowledge or reasoning. They allow the agent to interact with the outside world, perform calculations, fetch real-time data, or execute specific actions.

tools (Optional): Provide a list of tools the agent can use. Each item in the list can be:
A Python function (automatically wrapped as a FunctionTool).
An instance of a class inheriting from BaseTool.
An instance of another agent (AgentTool, enabling agent-to-agent delegation - see Multi-Agents).
The LLM uses the function/tool names, descriptions (from docstrings or the description field), and parameter schemas to decide which tool to call based on the conversation and its instructions.


# Define a tool function
def get_capital_city(country: str) -> str:
  """Retrieves the capital city for a given country."""
  # Replace with actual logic (e.g., API call, database lookup)
  capitals = {"france": "Paris", "japan": "Tokyo", "canada": "Ottawa"}
  return capitals.get(country.lower(), f"Sorry, I don't know the capital of {country}.")

# Add the tool to the agent
capital_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="capital_agent",
    description="Answers user questions about the capital city of a given country.",
    instruction="""You are an agent that provides the capital city of a country... (previous instruction text)""",
    tools=[get_capital_city] # Provide the function directly
)
Learn more about Tools in the Tools section.

Advanced Configuration & Control¶
Beyond the core parameters, LlmAgent offers several options for finer control:

Fine-Tuning LLM Generation (generate_content_config)¶
You can adjust how the underlying LLM generates responses using generate_content_config.

generate_content_config (Optional): Pass an instance of google.genai.types.GenerateContentConfig to control parameters like temperature (randomness), max_output_tokens (response length), top_p, top_k, and safety settings.


from google.genai import types

agent = LlmAgent(
    # ... other params
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2, # More deterministic output
        max_output_tokens=250
    )
)
Structuring Data (input_schema, output_schema, output_key)¶
For scenarios requiring structured data exchange, you can use Pydantic models.

input_schema (Optional): Define a Pydantic BaseModel class representing the expected input structure. If set, the user message content passed to this agent must be a JSON string conforming to this schema. Your instructions should guide the user or preceding agent accordingly.

output_schema (Optional): Define a Pydantic BaseModel class representing the desired output structure. If set, the agent's final response must be a JSON string conforming to this schema.

Constraint: Using output_schema enables controlled generation within the LLM but disables the agent's ability to use tools or transfer control to other agents. Your instructions must guide the LLM to produce JSON matching the schema directly.
output_key (Optional): Provide a string key. If set, the text content of the agent's final response will be automatically saved to the session's state dictionary under this key (e.g., session.state[output_key] = agent_response_text). This is useful for passing results between agents or steps in a workflow.


from pydantic import BaseModel, Field

class CapitalOutput(BaseModel):
    capital: str = Field(description="The capital of the country.")

structured_capital_agent = LlmAgent(
    # ... name, model, description
    instruction="""You are a Capital Information Agent. Given a country, respond ONLY with a JSON object containing the capital. Format: {"capital": "capital_name"}""",
    output_schema=CapitalOutput, # Enforce JSON output
    output_key="found_capital"  # Store result in state['found_capital']
    # Cannot use tools=[get_capital_city] effectively here
)
Managing Context (include_contents)¶
Control whether the agent receives the prior conversation history.

include_contents (Optional, Default: 'default'): Determines if the contents (history) are sent to the LLM.

'default': The agent receives the relevant conversation history.
'none': The agent receives no prior contents. It operates based solely on its current instruction and any input provided in the current turn (useful for stateless tasks or enforcing specific contexts).

stateless_agent = LlmAgent(
    # ... other params
    include_contents='none'
)
Planning & Code Execution¶
For more complex reasoning involving multiple steps or executing code:

planner (Optional): Assign a BasePlanner instance to enable multi-step reasoning and planning before execution. (See Multi-Agents patterns).
code_executor (Optional): Provide a BaseCodeExecutor instance to allow the agent to execute code blocks (e.g., Python) found in the LLM's response. (See Tools/Built-in tools).
Putting It Together: Example¶
Code
Here's the complete basic capital_agent:


# Full example code for the basic capital agent
# --- Full example code demonstrating LlmAgent with Tools vs. Output Schema ---
import json # Needed for pretty printing dicts

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

# --- 1. Define Constants ---
APP_NAME = "agent_comparison_app"
USER_ID = "test_user_456"
SESSION_ID_TOOL_AGENT = "session_tool_agent_xyz"
SESSION_ID_SCHEMA_AGENT = "session_schema_agent_xyz"
MODEL_NAME = "gemini-2.0-flash"

# --- 2. Define Schemas ---

# Input schema used by both agents
class CountryInput(BaseModel):
    country: str = Field(description="The country to get information about.")

# Output schema ONLY for the second agent
class CapitalInfoOutput(BaseModel):
    capital: str = Field(description="The capital city of the country.")
    # Note: Population is illustrative; the LLM will infer or estimate this
    # as it cannot use tools when output_schema is set.
    population_estimate: str = Field(description="An estimated population of the capital city.")

# --- 3. Define the Tool (Only for the first agent) ---
def get_capital_city(country: str) -> str:
    """Retrieves the capital city of a given country."""
    print(f"\n-- Tool Call: get_capital_city(country='{country}') --")
    country_capitals = {
        "united states": "Washington, D.C.",
        "canada": "Ottawa",
        "france": "Paris",
        "japan": "Tokyo",
    }
    result = country_capitals.get(country.lower(), f"Sorry, I couldn't find the capital for {country}.")
    print(f"-- Tool Result: '{result}' --")
    return result

# --- 4. Configure Agents ---

# Agent 1: Uses a tool and output_key
capital_agent_with_tool = LlmAgent(
    model=MODEL_NAME,
    name="capital_agent_tool",
    description="Retrieves the capital city using a specific tool.",
    instruction="""You are a helpful agent that provides the capital city of a country using a tool.
The user will provide the country name in a JSON format like {"country": "country_name"}.
1. Extract the country name.
2. Use the `get_capital_city` tool to find the capital.
3. Respond clearly to the user, stating the capital city found by the tool.
""",
    tools=[get_capital_city],
    input_schema=CountryInput,
    output_key="capital_tool_result", # Store final text response
)

# Agent 2: Uses output_schema (NO tools possible)
structured_info_agent_schema = LlmAgent(
    model=MODEL_NAME,
    name="structured_info_agent_schema",
    description="Provides capital and estimated population in a specific JSON format.",
    instruction=f"""You are an agent that provides country information.
The user will provide the country name in a JSON format like {{"country": "country_name"}}.
Respond ONLY with a JSON object matching this exact schema:
{json.dumps(CapitalInfoOutput.model_json_schema(), indent=2)}
Use your knowledge to determine the capital and estimate the population. Do not use any tools.
""",
    # *** NO tools parameter here - using output_schema prevents tool use ***
    input_schema=CountryInput,
    output_schema=CapitalInfoOutput, # Enforce JSON output structure
    output_key="structured_info_result", # Store final JSON response
)

# --- 5. Set up Session Management and Runners ---
session_service = InMemorySessionService()

# Create separate sessions for clarity, though not strictly necessary if context is managed
session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID_TOOL_AGENT)
session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID_SCHEMA_AGENT)

# Create a runner for EACH agent
capital_runner = Runner(
    agent=capital_agent_with_tool,
    app_name=APP_NAME,
    session_service=session_service
)
structured_runner = Runner(
    agent=structured_info_agent_schema,
    app_name=APP_NAME,
    session_service=session_service
)

# --- 6. Define Agent Interaction Logic ---
async def call_agent_and_print(
    runner_instance: Runner,
    agent_instance: LlmAgent,
    session_id: str,
    query_json: str
):
    """Sends a query to the specified agent/runner and prints results."""
    print(f"\n>>> Calling Agent: '{agent_instance.name}' | Query: {query_json}")

    user_content = types.Content(role='user', parts=[types.Part(text=query_json)])

    final_response_content = "No final response received."
    async for event in runner_instance.run_async(user_id=USER_ID, session_id=session_id, new_message=user_content):
        # print(f"Event: {event.type}, Author: {event.author}") # Uncomment for detailed logging
        if event.is_final_response() and event.content and event.content.parts:
            # For output_schema, the content is the JSON string itself
            final_response_content = event.content.parts[0].text

    print(f"<<< Agent '{agent_instance.name}' Response: {final_response_content}")

    current_session = session_service.get_session(app_name=APP_NAME,
                                                  user_id=USER_ID,
                                                  session_id=session_id)
    stored_output = current_session.state.get(agent_instance.output_key)

    # Pretty print if the stored output looks like JSON (likely from output_schema)
    print(f"--- Session State ['{agent_instance.output_key}']: ", end="")
    try:
        # Attempt to parse and pretty print if it's JSON
        parsed_output = json.loads(stored_output)
        print(json.dumps(parsed_output, indent=2))
    except (json.JSONDecodeError, TypeError):
         # Otherwise, print as string
        print(stored_output)
    print("-" * 30)


# --- 7. Run Interactions ---
async def main():
    print("--- Testing Agent with Tool ---")
    await call_agent_and_print(capital_runner, capital_agent_with_tool, SESSION_ID_TOOL_AGENT, '{"country": "France"}')
    await call_agent_and_print(capital_runner, capital_agent_with_tool, SESSION_ID_TOOL_AGENT, '{"country": "Canada"}')

    print("\n\n--- Testing Agent with Output Schema (No Tool Use) ---")
    await call_agent_and_print(structured_runner, structured_info_agent_schema, SESSION_ID_SCHEMA_AGENT, '{"country": "France"}')
    await call_agent_and_print(structured_runner, structured_info_agent_schema, SESSION_ID_SCHEMA_AGENT, '{"country": "Japan"}')

if __name__ == "__main__":
    await main()
(This example demonstrates the core concepts. More complex agents might incorporate schemas, context control, planning, etc.)