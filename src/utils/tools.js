// src/utils/tools.js
// Tool definitions must mirror Google ADK Python exports exactly

// Old imports replaced with new unified path and class names
import {
  google_search_tool_instance,
  code_execution_tool_instance,
  AgentTool,
  FileTool,
  HttpRequestTool,
} from 'google-adk';

export const builtInTools = {
  googleSearch: {
    id: google_search_tool_instance.id,
    run: google_search_tool_instance.run,
    // adjust param schema if changed upstream
    params: google_search_tool_instance.params,
  },
  codeExecution: {
    id: code_execution_tool_instance.id,
    run: code_execution_tool_instance.run,
    params: code_execution_tool_instance.params,
  },
  httpRequest: {
    id: HttpRequestTool.id,
    run: HttpRequestTool.run,
    params: HttpRequestTool.params,
  },
  file: {
    id: FileTool.id,
    run: FileTool.run,
    params: FileTool.params,
  },
};

// --- Example Function Tools ---
export const exampleFunctionTools = [
  {
    id: 'stockprice',
    name: 'get_stock_price',
    description: 'Retrieves the current stock price for a given symbol using yfinance.',
    category: 'Function',
    parameters: {
      symbol: { type: 'string', description: 'The stock symbol (e.g., "AAPL", "GOOG"). Required.' }
    },
    return_info: "Returns a dictionary: {'result': <price_float_or_None>}",
    requirements: "`pip install yfinance`",
    code: `import yfinance as yf
import json

# Note: ADK automatically wraps Python functions added to the tools list.
def get_stock_price(symbol: str) -> float | None:
    """
    Retrieves the current closing stock price for a given symbol using yfinance.

    Args:
        symbol (str): The stock symbol (e.g., "AAPL", "GOOG").

    Returns:
        float: The current stock price as a float.
        None: If the symbol is invalid or data cannot be retrieved.
    """
    # ... (implementation remains the same) ...

# --- ADK Agent Setup (Example) ---
# from google.adk.agents import LlmAgent # Corrected import
# agent = LlmAgent(..., tools=[get_stock_price]) # Function added directly
`
  },
   {
    id: 'long_approval',
    name: 'ask_for_approval',
    description: 'Simulates asking a manager for approval (Long Running).',
    category: 'Long Running Function',
    parameters: {
      purpose: { type: 'string', description: 'Reason for the request.' },
      amount: { type: 'float', description: 'Amount requested.' }
    },
    return_info: "Initially returns {'status': 'pending', ...}. Final update sent later.",
    requirements: "Requires Agent Client logic to handle intermediate/final responses.",
    code: `from google.adk.tools import LongRunningFunctionTool
from google.adk.tools import ToolContext
from typing import Any, Dict

# 1. Define the long running function
def ask_for_approval(purpose: str, amount: float) -> Dict[str, Any]:
    """Simulates asking for approval and returns initial pending status."""
    # ... (implementation remains the same) ...

# 2. Wrap the function
approval_tool = LongRunningFunctionTool(func=ask_for_approval) # Show wrapping

# --- ADK Agent Setup (Example) ---
# from google.adk.agents import LlmAgent # Corrected import
# agent = LlmAgent(..., tools=[approval_tool]) # Add the *wrapped* tool
`
  },
];

// --- Example Agent-as-Tool Reference ---
export const exampleAgentAsTool = {
    id: 'summary_agent_ref',
    name: 'summarize_text',
    description: 'Summarizes long passages of text by calling the dedicated Summary Agent.',
    category: 'Agent-as-Tool',
    targetAgentName: 'summary_agent',
    requirements: "Requires 'summary_agent' to be defined and runnable.",
    // Corrected: Show AgentTool instantiation
    code_example: `from google.adk.agents import LlmAgent # Corrected import
from google.adk.tools import AgentTool

# Assume summary_agent is an instance of LlmAgent defined elsewhere
# from .summary_agent_def import summary_agent

# Define the tool that calls the other agent
summarizer_tool = AgentTool(agent=summary_agent, name="summarize_text") # Explicitly name the tool call

# Add it to the parent agent's tools
parent_agent = LlmAgent(
    # ... other parent agent params
    instruction="...If you need to summarize text, use the 'summarize_text' tool...",
    tools=[summarizer_tool]
)`
};

export function wrapAsAgentTool(toolInstance) {
  // Upstream now expects AgentTool({ tool: instance }) instead of old signature
  return new AgentTool({ tool: toolInstance });
}

export const allTools = [
    ...Object.values(builtInTools),
    ...exampleFunctionTools,
    // { ...exampleAgentAsTool, id: 'agent_tool_example' } // Example reference
];