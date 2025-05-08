// src/utils/codeTemplates.js

// --- Agent Definition Template (Based on ADK structure) ---
export const agentTemplate = `import datetime
# --- Core ADK Imports ---
from google.adk.agents import LlmAgent # Use LlmAgent
from google.adk.tools import ToolContext, AgentTool, BaseTool # Base classes
from google.adk.agents.callback_context import CallbackContext # For callbacks
from google.genai import types as genai_types # For GenerateContentConfig

# --- Tool Imports ---
# Import built-in tools used (references, not instantiated here)
{{tool_imports}}
# Import custom tool functions/classes if defined in separate files
# from .my_custom_tools import my_custom_tool_function

# --- Sub-Agent Imports ---
# Import other LlmAgent instances if using AgentTool
# from .sub_agents import summary_agent_instance, db_agent_instance

# --- Define Custom Tools (if not imported) ---
{{custom_tool_definitions}}

# --- Define Agent Callbacks (Optional) ---
# Example: Modify LLM request before sending
def before_model_callback_example(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    print(f"*** Intercepting LLM Request for agent: {callback_context.agent_name} ***")
    # Example: Add a safety preamble
    # llm_request.append_instructions(["Always respond safely and ethically."])
    return None # Return None to proceed with the call

# Example: Log tool calls
def before_tool_callback_example(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
) -> Optional[dict]:
     print(f"*** Agent '{tool_context.agent_name}' calling tool '{tool.name}' with args: {args} ***")
     return None # Return None to execute the actual tool

# --- Instantiate Tools ---
# Built-in tools are often added directly by reference if no config needed
# Custom functions are added directly by reference
# Tools requiring config (VertexAiSearchTool, AgentTool) are instantiated
tools_list = [
    {{tools_instantiation}}
]

# --- Define the Main Agent ---
agent = LlmAgent(
    name="{{name}}",
    model="{{model}}",
    instruction="""{{instructions}}""", # Instructions guide behavior
    description="{{description}}", # Used when called as a tool by another agent
    tools=tools_list,
    # --- Model Generation Configuration ---
    generate_content_config=genai_types.GenerateContentConfig(
        temperature={{temperature}},
        top_p={{topP}},
        top_k={{topK}},
        max_output_tokens={{maxOutputTokens}},
        # Add safety_settings, stop_sequences if needed
    ),
    # --- Optional: Structured Output ---
    # output_schema=MyOutputSchema, # Requires defining MyOutputSchema (Pydantic model)
    # output_key="my_result_key", # Store result in state['my_result_key']

    # --- Optional: Callbacks ---
    # before_model_callback=before_model_callback_example,
    # before_tool_callback=before_tool_callback_example,
    # Add other callbacks: after_model_callback, after_tool_callback, etc.

    # --- Optional: Other LlmAgent parameters ---
    # include_contents='none', # To make agent stateless regarding history
    # planner=MyPlanner(),
    # code_executor=MyCodeExecutor(),
)

print(f"Agent '{agent.name}' created successfully.")

# To run this agent, use the ADK Runner
# See runnerTemplate and streamingTemplate examples
`;

// --- Custom Tool Template (Standard Python Function) ---
export const functionToolTemplate = `# Define your Python function
# Use type hints for parameters and return value
# The docstring is crucial for the LLM to understand the tool

def {{name}}({{parameters}}) -> {{return_type}}:
    \"\"\"{{description}}

    Args:
        {{args_docs}}

    Returns:
        {{return_doc}}
    \"\"\"
    # Function implementation
    print(f"Executing tool: {{name}} with args: {{args_list}}")
    # Replace with actual logic
    result = "Simulated result from {{name}}"
    return result # Return value should be JSON serializable (dict preferred)

# --- ADK Agent Setup ---
# from google.adk.agents import LlmAgent
#
# # Add the function *directly* to the agent's tools list
# agent = LlmAgent(
#     # ... other agent params
#     tools=[{{name}}] # Pass the function object
# )
`;

// --- Structured Output Template (Using Pydantic) ---
export const structuredOutputTemplate = `from pydantic import BaseModel, Field
from typing import List, Optional

# Define your Pydantic model for the desired output structure
class {{className}}(BaseModel):
    \"\"\"{{class_description}}\"\"\"
    {{fields}}

# --- ADK Agent Setup ---
# from google.adk.agents import LlmAgent
#
# agent = LlmAgent(
#     name="{{name}}",
#     model="{{model}}",
#     instruction="""{{instructions}}
#
#     **IMPORTANT**: Your final response MUST be a JSON object conforming exactly
#     to the {{className}} schema. Do not add any introductory text or explanations
#     outside the JSON structure.
#     """,
#     output_schema={{className}}, # Set the output schema
#     # Note: Setting output_schema disables tool usage for this agent.
# )
`;

// --- Guardrail/Callback Template ---
export const guardrailTemplate = `# ADK uses Callbacks for guardrails and lifecycle hooks.
# Define Python functions to be registered with the agent.

from google.adk.agents import LlmAgent, BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools import BaseTool, ToolContext
from google.genai import types
from typing import Optional, Any, Dict

# Example: Input Guardrail using before_model_callback
def block_harmful_input(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    \"\"\"Checks the last user message for prohibited content.\"\"\"
    last_user_message = ""
    if llm_request.contents and llm_request.contents[-1].role == 'user':
         if llm_request.contents[-1].parts:
            last_user_message = llm_request.contents[-1].parts[0].text or ""

    print(f"[Callback] Checking input: '{last_user_message[:50]}...'")
    if "harmful keyword" in last_user_message.lower():
        print("[Callback] Harmful content detected! Blocking LLM call.")
        # Return an LlmResponse to skip the LLM call and provide a fixed response
        return LlmResponse(
            content=types.Content(role="model", parts=[
                types.Part(text="I cannot process requests containing harmful content.")
            ])
        )
    # Return None to allow the LLM call to proceed
    return None

# Example: Output Guardrail using after_model_callback
def redact_pii_output(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    \"\"\"Redacts potential PII (like SSN) from the model's response.\"\"\"
    if llm_response.content and llm_response.content.parts:
        modified = False
        for part in llm_response.content.parts:
            if part.text and "SSN:" in part.text:
                 print("[Callback] Redacting PII from output.")
                 # Basic redaction example - use more robust methods in production
                 part.text = part.text.replace("SSN:", "[REDACTED SSN]:")
                 modified = True
        # Return the modified response if changes were made
        return llm_response if modified else None
    # Return None to use the original response
    return None

# --- ADK Agent Setup ---
# agent = LlmAgent(
#     name="{{name}}",
#     model="{{model}}",
#     instruction="{{instructions}}",
#     # Register callbacks
#     before_model_callback=block_harmful_input,
#     after_model_callback=redact_pii_output,
# )
`;

// --- Runner Template (Based on ADK Runner.run_async) ---
export const runnerTemplate = `import asyncio
import json
import os
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService # Or DatabaseSessionService, etc.
from google.genai import types

# Assuming the agent definition is in agent.py
from agent import agent

# --- Session and Runner Setup ---
APP_NAME = "{{name}}_app" # Use agent name or define app name
USER_ID = "test_user_1"
SESSION_ID = "session_1"

async def main():
    # Ensure Google Cloud credentials are set in the environment
    # export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json" OR export GOOGLE_API_KEY="..."
    # export GOOGLE_PROJECT_ID="..." (if using Vertex AI)

    session_service = InMemorySessionService() # Use a persistent service in production
    session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

    user_prompt = "{{prompt}}"
    print(f"--- Running Agent '{agent.name}' for Prompt: '{user_prompt}' ---")

    user_message = types.Content(role='user', parts=[types.Part(text=user_prompt)])

    try:
        final_response_text = "Agent execution finished without a final text response."
        async for event in runner.run_async(user_id=USER_ID, session_id=session.id, new_message=user_message):
            print(f"Event ID: {event.id}, Author: {event.author}") # Basic event logging
            # Process specific event parts if needed (e.g., tool calls/responses)
            if event.content and event.content.parts:
                 for part in event.content.parts:
                      if part.function_call:
                           print(f"  Tool Call: {part.function_call.name}({part.function_call.args})")
                      elif part.function_response:
                           print(f"  Tool Response ({part.function_response.name}): {part.function_response.response}")
                      elif part.text:
                           print(f"  Text: '{part.text}'")

            # Capture the final text response
            if event.is_final_response() and event.content and event.content.parts:
                 if event.content.parts[0].text:
                      final_response_text = event.content.parts[0].text
                 else:
                      # Handle non-text final response (e.g., structured output)
                      try:
                           final_response_text = json.dumps(event.content.parts[0].model_dump(exclude_none=True))
                      except Exception:
                           final_response_text = "[Non-text final response]"

        print("\\n--- Final Agent Response ---")
        print(final_response_text)

    except Exception as e:
        print(f"\\n--- Error running agent ---")
        print(e)
        import traceback
        traceback.print_exc()

    print("--------------------------\\n")

if __name__ == "__main__":
  asyncio.run(main())
`;

// --- Streaming Template (Based on ADK Runner.run_async with streaming config) ---
export const streamingTemplate = `import asyncio
import json
import os
from google.adk.agents import RunConfig, StreamingMode # Import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Assuming the agent definition is in agent.py
from agent import agent

# --- Session and Runner Setup ---
APP_NAME = "{{name}}_stream_app"
USER_ID = "test_user_stream_1"
SESSION_ID = "session_stream_1"

async def main():
    # Ensure Google Cloud credentials are set
    session_service = InMemorySessionService()
    session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

    user_prompt = "{{prompt}}"
    print(f"--- Streaming Agent '{agent.name}' for Prompt: '{user_prompt}' ---")

    user_message = types.Content(role='user', parts=[types.Part(text=user_prompt)])

    # Configure for streaming
    run_config = RunConfig(streaming_mode=StreamingMode.SSE) # Use SSE for server-sent events

    try:
        print("\\n--- Agent Stream Events ---")
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session.id,
            new_message=user_message,
            run_config=run_config # Pass the streaming config
        ):
            print(f"Event ID: {event.id}, Author: {event.author}, Partial: {event.partial}")
            if event.content and event.content.parts:
                 for part in event.content.parts:
                      if part.text:
                           print(f"  Text Chunk: '{part.text}'")
                      elif part.function_call:
                           print(f"  Tool Call: {part.function_call.name}({part.function_call.args})")
                      elif part.function_response:
                           print(f"  Tool Response ({part.function_response.name}): {part.function_response.response}")
            if event.is_final_response():
                 print("--- Final Response Received ---")


    except Exception as e:
        print(f"\\n--- Error during agent stream ---")
        print(e)
        import traceback
        traceback.print_exc()

    print("\\n--------------------------\\n")

if __name__ == "__main__":
  asyncio.run(main())
`;

// Template for a full custom tool example (standard Python function)
export const fullToolExampleTemplate = `import json
from typing import Dict, Any, List # Use standard typing

# Define the Python function
# Use type hints and a clear docstring
def get_stock_price(ticker_symbol: str) -> str:
    \"\"\"Fetches the current stock price for a given ticker symbol.

    Args:
        ticker_symbol: The stock ticker symbol (e.g., 'GOOGL').

    Returns:
        A JSON string containing the ticker and its simulated price, or an error message.
    \"\"\"
    print(f"Executing tool: get_stock_price with symbol: {ticker_symbol}")
    # --- Placeholder Implementation ---
    # In a real scenario, this would call a financial data API.
    ticker_symbol = ticker_symbol.upper()
    simulated_prices = {
        "GOOGL": 175.20,
        "AAPL": 190.50,
        "MSFT": 410.80,
    }
    price = simulated_prices.get(ticker_symbol)

    if price:
        # Return a dictionary (preferred by ADK), it will be JSON serialized
        return {"ticker": ticker_symbol, "price": price, "currency": "USD"}
    else:
        return {"error": f"Could not find price for ticker: {ticker_symbol}"}

# --- ADK Agent Setup ---
# from google.adk.agents import LlmAgent
#
# # Add the function directly to the agent's tools list
# agent = LlmAgent(
#     model="gemini-2.5-flash-preview-04-17",
#     instruction="Use the get_stock_price tool when asked for stock prices.",
#     tools=[get_stock_price] # Pass the function object
# )
`;