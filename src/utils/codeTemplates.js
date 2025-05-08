// src/utils/codeTemplates.js

// --- Agent Definition Template (Based on ADK structure assumptions) ---
export const agentTemplate = `import datetime
# --- Core ADK Imports ---
from google.adk import Agent, ToolContext, AgentTool, BaseTool # Adjust imports as needed
from google.adk.agents import Callback, CallbackContextMenu # Example for callbacks

# --- Tool Imports ---
# Import built-in tools used
{{tool_imports}}
# Import custom tool functions/classes if defined in separate files
# from .my_custom_tools import my_custom_tool_function

# --- Sub-Agent Imports ---
# Import sub-agent definitions if used
# from .sub_agents.billing_agent import billing_agent_definition

# --- Define Custom Tools (if not imported) ---
{{custom_tool_definitions}}

# --- Define Agent Callbacks (Optional) ---
# Example: Setup before agent call
def setup_before_agent_call(context: CallbackContextMenu) -> None:
    """Sets up initial state or context before the agent runs."""
    print(f"*** Setting up agent '{context.agent.name}' ***")
    # Example: Injecting database schema (if applicable and loaded)
    # if context.agent.name == 'database_agent' and 'db_schema' in loaded_artifacts:
    #    context.state['db_schema'] = loaded_artifacts['db_schema']
    context.state['current_date'] = datetime.date.today().isoformat()
    print(f"Current date set in state: {context.state['current_date']}")

# --- Instantiate Tools ---
tools_list = [
    {{tools_instantiation}}
]

# --- Instantiate Sub-Agents (if applicable) ---
sub_agents_list = [
    {{sub_agents_instantiation}}
]

# --- Define the Main Agent ---
agent = GoogleAgent(
    name="{{name}}",
    instructions="""{{instructions}}

    Current date: {{current_date}}""", # Example of injecting state
    model="{{model}}",
    tools=tools_list,
    # sub_agents=sub_agents_list, # Use if ADK supports sub_agents parameter
    # --- Model Configuration (Example - adjust based on ADK) ---
    # model_config={
    #    "temperature": {{temperature}},
    #    "top_p": {{topP}},
    #    "top_k": {{topK}},
    #    "max_output_tokens": {{maxOutputTokens}},
    # },
    # --- Callbacks (Example) ---
    # callbacks=[
    #     Callback(before_agent_call=setup_before_agent_call),
    # ]
    # --- Other ADK specific parameters ---
)

# --- Optional: Define Agent Tree (if using sub_agents) ---
# Example structure, depends heavily on ADK's multi-agent approach
# root_agent = GoogleAgent(
#     name="root_{{name}}",
#     model="{{model}}",
#     instructions="Orchestrate tasks between sub-agents.",
#     sub_agents=sub_agents_list # Pass instantiated sub-agents here
# )

print(f"Agent '{agent.name}' created successfully.")

# To run this agent, you would typically use a runner provided by the ADK
# Example (adjust based on actual ADK runner):
#
# import asyncio
# from google.adk import run_agent_async
#
# async def main():
#   user_query = "What are the total sales for Canada?"
#   print(f"\\n--- Running Agent for Query: '{user_query}' ---")
#   result = await run_agent_async(agent, user_query) # Or root_agent if using a tree
#   print("\\n--- Agent Result ---")
#   print(result)
#   print("--------------------\\n")
#
# if __name__ == "__main__":
#   asyncio.run(main())
`;

// --- Custom Tool Template (Standard Python Function) ---
export const functionToolTemplate = `# Potential ADK tool decorator (if applicable)
# from google.adk.tools import tool

# @tool
def {{name}}({{parameters}}) -> {{return_type}}: # Specify return type hint
    \"\"\"{{description}}

    Args:
        {{args_docs}}

    Returns:
        {{return_doc}}
    \"\"\"
    # Function implementation goes here
    print(f"Executing tool: {{name}} with args: {{args_list}}")
    # Replace with actual logic
    result = "Simulated result from {{name}}"
    return result
`;

// --- Structured Output Template (Using Pydantic - common pattern) ---
// ADK might have its own mechanism or support Pydantic directly.
export const structuredOutputTemplate = `from pydantic import BaseModel, Field
from typing import List, Optional

class {{className}}(BaseModel):
    \"\"\"{{class_description}}\"\"\"
    {{fields}}

# --- ADK Agent Setup (Example) ---
# from google.adk import Agent
#
# agent = Agent(
#     name="{{name}}",
#     instructions="""{{instructions}}
#
#     Respond using the {{className}} structure.""",
#     output_schema={{className}}, # Hypothetical ADK parameter for structured output
#     model="{{model}}"
# )
`;

// --- Guardrail Template (Placeholder - Highly ADK Dependent) ---
export const guardrailTemplate = `# Guardrail implementation depends heavily on the ADK's specific mechanisms.
# It might involve callbacks, specific guardrail classes, or other patterns.

# --- Example using Hypothetical Callbacks ---
# from google.adk.agents import Callback, CallbackContextMenu, GuardrailViolation

# def input_content_filter(context: CallbackContextMenu) -> None:
#     user_input = context.request # Assuming request holds user input
#     print(f"Applying input guardrail to: {user_input}")
#     if "prohibited content" in user_input.lower():
#         # ADK might have a specific way to signal a violation
#         raise GuardrailViolation("Input contains prohibited content.")
#     print("Input guardrail passed.")

# def output_pii_filter(context: CallbackContextMenu) -> None:
#     agent_response = context.response # Assuming response holds agent output
#     print(f"Applying output guardrail to: {agent_response}")
#     if "SSN:" in agent_response:
#         # Modify response or raise violation
#         context.response = "[REDACTED PII]"
#         print("Output guardrail modified response.")
#         # Or: raise GuardrailViolation("Output contains PII.")
#     print("Output guardrail passed.")


# --- ADK Agent Setup (Example) ---
# from google.adk import Agent
#
# agent = Agent(
#     name="{{name}}",
#     instructions="{{instructions}}",
#     model="{{model}}",
#     callbacks=[
#         Callback(before_llm_call=input_content_filter), # Example hook point
#         Callback(after_llm_call=output_pii_filter)     # Example hook point
#     ]
# )
`;

// --- Runner Template (Placeholder - Based on Backend/ADK Runner) ---
export const runnerTemplate = `import asyncio
import json
# Assuming the agent definition is in agent.py
from agent import agent # Or potentially root_agent if using an agent tree

# --- Google ADK Runner Import (Hypothetical) ---
# This will depend on the actual ADK structure
try:
    from google.adk import run_agent_async
except ImportError:
    print("Google ADK not found. Cannot run agent.")
    async def run_agent_async(*args, **kwargs): return {"final_output": "ADK not installed - Simulated response"}

async def main():
    user_prompt = "{{prompt}}"
    print(f"--- Running Agent for Prompt: '{user_prompt}' ---")

    # Running the agent (adjust based on actual ADK runner function)
    # It might take the agent object and the prompt string.
    # Context might be handled implicitly or need explicit creation.
    try:
        result = await run_agent_async(agent, user_prompt)

        print("\\n--- Agent Result ---")
        # Pretty print if the result is complex (e.g., dict/JSON)
        if isinstance(result, (dict, list)):
             print(json.dumps(result, indent=2))
        else:
             print(result)

    except Exception as e:
        print(f"\\n--- Error running agent ---")
        print(e)

    print("--------------------\\n")

if __name__ == "__main__":
  # Ensure Google Cloud credentials are set in the environment
  # e.g., export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
  # or export GOOGLE_API_KEY="your-key"
  asyncio.run(main())
`;

// --- Streaming Template (Placeholder - Based on Backend/ADK Runner) ---
export const streamingTemplate = `import asyncio
import json
# Assuming the agent definition is in agent.py
from agent import agent # Or potentially root_agent

# --- Google ADK Streaming Runner Import (Hypothetical) ---
# This will depend heavily on the actual ADK streaming implementation
try:
    # Option 1: A dedicated streaming runner
    # from google.adk import run_agent_streamed
    # Option 2: The standard runner might return a streamable object
    from google.adk import run_agent_async
    # Assume run_agent_async returns an object with an async iterator method
except ImportError:
    print("Google ADK not found. Cannot run agent stream.")
    async def run_agent_async(*args, **kwargs):
        class MockStream:
             async def stream_events(self):
                 yield {"type": "final_output", "data": "ADK not installed - Simulated streaming response"}
        return MockStream()


async def main():
    user_prompt = "{{prompt}}"
    print(f"--- Streaming Agent for Prompt: '{user_prompt}' ---")

    try:
        # --- Running the Agent Stream ---
        # Adjust based on actual ADK streaming API
        run_result = await run_agent_async(agent, user_prompt) # Or run_agent_streamed

        print("\\n--- Agent Stream Events ---")
        # Assuming the result object has an async iterator for events
        async for event in run_result.stream_events(): # Hypothetical method name
            print(f"Event Type: {event.get('type')}")
            # Process different event types (highly ADK specific)
            if event.get('type') == 'token_delta': # Example: Token streaming
                print(event.get('data', ''), end="", flush=True)
            elif event.get('type') == 'tool_call': # Example: Tool call start
                print(f"\\nTool Call: {event.get('data', {}).get('name')}")
            elif event.get('type') == 'tool_result': # Example: Tool call end
                print(f"Tool Result: {event.get('data')}")
            elif event.get('type') == 'final_output': # Example: Final message
                print(f"\\nFinal Output: {event.get('data')}")
            else:
                # Print raw event for unknown types
                print(f"Raw Event: {json.dumps(event, indent=2)}")

    except Exception as e:
        print(f"\\n--- Error during agent stream ---")
        print(e)

    print("\\n-------------------------\\n")


if __name__ == "__main__":
  # Ensure Google Cloud credentials are set
  asyncio.run(main())
`;

// Template for a full custom tool example (standard Python)
export const fullToolExampleTemplate = `import json
from typing import Dict, Any, List # Use standard typing

# Potential ADK tool decorator (if applicable)
# from google.adk.tools import tool

# @tool
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
    # Ensure proper error handling and API key management.
    ticker_symbol = ticker_symbol.upper()
    simulated_prices = {
        "GOOGL": 175.20,
        "AAPL": 190.50,
        "MSFT": 410.80,
    }
    price = simulated_prices.get(ticker_symbol)

    if price:
        return json.dumps({"ticker": ticker_symbol, "price": price, "currency": "USD"})
    else:
        return json.dumps({"error": f"Could not find price for ticker: {ticker_symbol}"})

# --- ADK Agent Setup (Example) ---
# from google.adk import Agent
# stock_price_tool = get_stock_price
# agent = Agent(..., tools=[stock_price_tool])
`;