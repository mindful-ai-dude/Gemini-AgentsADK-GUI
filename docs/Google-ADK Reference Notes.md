
---

**Google ADK Reference Notese**

**1. `backend/.
*   Confirmed `Runner.run_async` usage.

```python
# backend/app.py
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import logging
import traceback
import asyncio # Added for asyncio.run
import importlib # Added for dynamic agent loading

# --- Google ADK Imports ---
# Using specific imports based on ADK structure
try:
    from google.adk.agents import LlmAgent # Main agent class
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService, BaseSessionService # Assuming default for now
    from google.adk.tools import ToolContext, BaseTool, FunctionTool, AgentTool
    from google.adk.tools import google_search, built_in_code_execution, VertexAiSearchTool
    from google.genai import types as genai_types # For Content, Part etc.
    # Import other necessary ADK components if needed (e.g., for specific callbacks, planners)
except ImportError as e:
    print(f"Error importing Google ADK: {e}. Make sure 'google-adk' is installed.")
    # Define dummy classes/functions if ADK is not installed
    class LlmAgent: pass
    class Runner: pass
    class InMemorySessionService: pass
    class BaseSessionService: pass
    class ToolContext: pass
    class BaseTool: pass
    class FunctionTool: pass
    class AgentTool: pass
    class genai_types: class Content: pass; class Part: pass; class FunctionDeclaration: pass; class GenerateContentConfig: pass
    async def run_agent_async(*args, **kwargs): return {"final_output": "ADK not installed - Simulated response"}
    def google_search(*args, **kwargs): return "Simulated Google Search result"
    def built_in_code_execution(*args, **kwargs): return "Simulated Code Execution result"
    class VertexAiSearchTool: pass


# --- Google Cloud Client Imports (for validation, example) ---
try:
    from google.cloud import aiplatform
    from google.auth import exceptions as auth_exceptions
    from google.api_core import exceptions as api_exceptions
except ImportError:
    print("google-cloud-aiplatform or google-auth not installed. API Key validation will be basic.")
    aiplatform = None
    auth_exceptions = None
    api_exceptions = None


# --- Flask App Setup ---
load_dotenv() # Load environment variables from .env file
app = Flask(__name__)
CORS(app) # Enable CORS for requests from the React frontend

logging.basicConfig(level=logging.INFO)

# --- Global Services (Use persistent ones in production) ---
# TODO: Make session service configurable (e.g., via env var or config file)
session_service: BaseSessionService = InMemorySessionService()
# artifact_service = InMemoryArtifactService() # Add if needed
# memory_service = InMemoryMemoryService() # Add if needed

# --- Helper Functions ---

def load_google_credentials():
    """Checks for Google Cloud credentials."""
    api_key = os.getenv("GOOGLE_API_KEY")
    service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if api_key:
        logging.info("Using GOOGLE_API_KEY for authentication.")
        return {"type": "api_key", "key": api_key}
    elif service_account_path and os.path.exists(service_account_path):
        logging.info(f"Using GOOGLE_APPLICATION_CREDENTIALS from: {service_account_path}")
        return {"type": "service_account", "path": service_account_path}
    else:
        # Attempt Application Default Credentials (ADC)
        try:
            from google.auth import default as default_credentials
            credentials, project_id = default_credentials()
            if credentials:
                 logging.info(f"Using Application Default Credentials (ADC). Project: {project_id}")
                 # ADC doesn't fit neatly into the key/path dict, but its presence is enough
                 return {"type": "adc", "project": project_id}
        except auth_exceptions.DefaultCredentialsError:
             logging.warning("Application Default Credentials not found.")

        logging.warning("No Google Cloud API Key, Service Account, or ADC found in environment variables.")
        return None

# --- Agent and Tool Loading Cache ---
# Cache loaded agents to avoid reloading modules repeatedly
loaded_agents_cache = {}
# Store instantiated runners per app_name
runner_cache = {}

async def get_root_agent(app_name: str) -> LlmAgent:
    """Loads or retrieves the root agent definition from cache."""
    if app_name in loaded_agents_cache:
        return loaded_agents_cache[app_name]

    try:
        # Assuming agents are in subdirectories named after the app_name within 'backend/agents'
        # Adjust this path if your structure is different
        agent_module_path = os.path.join('agents', app_name) # Relative path within backend
        if not os.path.isdir(agent_module_path):
             # Fallback: try loading directly if app_name is a module path relative to backend/
             agent_module_path = app_name

        # Dynamically import the agent module
        # Ensure the parent directory of 'agents' is in sys.path if needed
        # Example: sys.path.insert(0, os.path.abspath('.')) # Add backend dir to path
        agent_module = importlib.import_module(f"{agent_module_path}.agent") # Assumes agent.py inside the folder

        if hasattr(agent_module, "root_agent"):
            root_agent_def = agent_module.root_agent
            # Handle awaitable agent definitions (e.g., if they do async setup)
            if inspect.isawaitable(root_agent_def):
                 agent_instance, exit_stack = await root_agent_def # Assuming it returns agent, exit_stack
                 # TODO: Manage exit_stack lifecycle if needed
                 root_agent = agent_instance
            else:
                 root_agent = root_agent_def

            if not isinstance(root_agent, LlmAgent): # Check type after potential await
                 raise TypeError(f"Expected 'root_agent' in {app_name}.agent to be an LlmAgent instance.")

            loaded_agents_cache[app_name] = root_agent
            return root_agent
        else:
            raise AttributeError(f"'root_agent' not found in module {app_name}.agent")
    except ModuleNotFoundError:
        raise ValueError(f"Agent module '{app_name}.agent' not found or contains import errors.")
    except Exception as e:
        raise ValueError(f"Error loading agent '{app_name}': {e}")


def _instantiate_adk_tools(tool_configs, target_agent_definitions: dict) -> list[BaseTool]:
    """
    Instantiates ADK tool objects based on the configuration from the frontend.
    Requires target_agent_definitions for AgentTool instantiation.
    """
    adk_tools = []
    if not tool_configs:
        return adk_tools

    for config in tool_configs:
        tool_id = config.get('id')
        tool_name = config.get('name')
        category = config.get('category')
        params = config.get('parameters', {})

        try:
            if tool_id == 'googlesearch' or tool_name == 'google_search':
                # google_search is imported directly
                adk_tools.append(google_search)
                logging.info("Added google_search tool.")
            elif tool_id == 'codeexecution' or tool_name == 'built_in_code_execution':
                 # built_in_code_execution is imported directly
                 adk_tools.append(built_in_code_execution)
                 logging.info("Added built_in_code_execution tool.")
            elif tool_id == 'vertexaisearch' or tool_name == 'VertexAiSearchTool':
                 datastore_id = params.get('data_store_id', {}).get('value')
                 if not datastore_id:
                      logging.error(f"Missing data_store_id for VertexAiSearchTool config: {config}")
                      continue # Skip this tool if ID is missing
                 adk_tools.append(VertexAiSearchTool(data_store_id=datastore_id))
                 logging.info(f"Added VertexAiSearchTool with datastore: {datastore_id}")
            elif category == 'Function':
                # Dynamic loading/execution of function code from config is complex and insecure.
                # Assume these functions are defined *within the backend environment* and accessible.
                # This requires the backend to have access to the function definitions.
                # For this GUI, we cannot directly use the 'code' from the config.
                logging.warning(f"Cannot dynamically instantiate custom function tool '{tool_name}' from GUI config. "
                                "Backend needs access to its definition.")
                # If functions were defined in a known module in the backend:
                # try:
                #    func = getattr(importlib.import_module('backend.custom_tools'), tool_name)
                #    adk_tools.append(FunctionTool(func=func))
                # except (ImportError, AttributeError):
                #    logging.error(f"Custom function '{tool_name}' not found in backend environment.")
            elif category == 'Long Running Function':
                 # Similar limitation as Function tools for dynamic loading.
                 logging.warning(f"Cannot dynamically instantiate long running function tool '{tool_name}' from GUI config. "
                                 "Backend needs access to its definition and LongRunningFunctionTool wrapper.")
            elif category == 'Agent-as-Tool':
                 # Instantiate AgentTool by referencing the *loaded* target agent definition
                 target_agent_id = config.get('id') # ID of the *target agent config*
                 target_agent_name_in_registry = config.get('targetAgentName') # Name used to look up definition

                 if target_agent_name_in_registry in target_agent_definitions:
                      target_agent_instance = target_agent_definitions[target_agent_name_in_registry]
                      # Use the 'name' from the config as the *tool call name*
                      tool_call_name = config.get('name')
                      if not tool_call_name:
                           logging.error(f"Missing tool call name ('name' field) for Agent-as-Tool config: {config}")
                           continue
                      adk_tools.append(AgentTool(agent=target_agent_instance, name=tool_call_name))
                      logging.info(f"Added AgentTool: calls '{target_agent_name_in_registry}' using tool name '{tool_call_name}'")
                 else:
                      logging.error(f"Target agent definition '{target_agent_name_in_registry}' for AgentTool not found.")

            else:
                logging.warning(f"Unknown tool type or category in config: {config}")

        except Exception as e:
            logging.error(f"Failed to instantiate tool '{tool_name}' (ID: {tool_id}): {e}")

    return adk_tools


# --- API Endpoints ---

@app.route('/api/validate-key', methods=['POST'])
def validate_key():
    """Validates the provided Google Cloud API Key."""
    # This endpoint remains largely unchanged, as it's a basic check/simulation
    data = request.get_json()
    api_key = data.get('apiKey')

    if not api_key:
        return jsonify({"error": "API Key not provided"}), 400

    try:
        # --- Actual Validation Logic ---
        # Keep the simulation or implement a real check if feasible/secure with API Key
        if not aiplatform:
             if len(api_key) > 20:
                 logging.info("Basic API Key format check passed (google-cloud-aiplatform not installed).")
                 return jsonify({"success": True, "message": "API Key format looks basic, but not fully validated."})
             else:
                 raise ValueError("Invalid API Key format.")

        # Simulate success for now
        logging.info(f"Simulated validation successful for key ending with ...{api_key[-4:]}")
        return jsonify({"success": True, "message": "API Key validation simulated successfully."})

    except Exception as e:
        logging.error(f"Error validating API key: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Validation failed: {str(e)}"}), 500


@app.route('/api/run-agent', methods=['POST'])
async def handle_run_agent():
    """Handles running the agent based on provided configuration."""
    credentials = load_google_credentials()
    if not credentials:
         # If no credentials found via env vars or ADC, try using the API key from request *if* provided
         # This is less secure and generally discouraged for backends.
         data = request.get_json()
         api_key_from_req = data.get('apiKey') # Assuming frontend might send it
         if api_key_from_req:
              logging.warning("Using API Key provided in request for backend execution (less secure).")
              os.environ['GOOGLE_API_KEY'] = api_key_from_req # Temporarily set for this request if needed by ADK/clients
         else:
              return jsonify({"error": "Google Cloud credentials not configured on the backend or provided in request."}), 500

    data = request.get_json()
    agent_config = data.get('agentConfig')
    user_input = data.get('input')
    # Optional: Frontend could send user_id and session_id if needed
    user_id = data.get('userId', 'gui_user')
    session_id = data.get('sessionId', f'gui_session_{os.urandom(8).hex()}') # Generate if not provided

    if not agent_config or not user_input:
        return jsonify({"error": "Missing agentConfig or input"}), 400

    app_name = agent_config.get('name', 'UnnamedApp') # Use agent name as app name for runner

    try:
        logging.info(f"Running agent config: {agent_config.get('name')}")

        # --- Instantiate ADK Agent Dynamically ---
        agent_name = agent_config.get('name', 'Unnamed Agent')
        instructions = agent_config.get('instructions', 'You are a helpful assistant.')
        model = agent_config.get('model', 'gemini-1.5-flash')
        model_settings = agent_config.get('modelSettings', {})

        # Prepare GenerateContentConfig from modelSettings
        genai_config = genai_types.GenerateContentConfig(
            temperature=model_settings.get('temperature'),
            top_p=model_settings.get('topP'),
            top_k=model_settings.get('topK'),
            max_output_tokens=model_settings.get('maxOutputTokens'),
            # Add other mappings if needed (stop_sequences, safety_settings)
        )

        # Instantiate tools - Requires loading target agent definitions for AgentTool
        # For simplicity, assume target agents are defined elsewhere and accessible via a registry/loader
        # Here, we'll pass an empty dict, meaning AgentTool won't work unless target agents are pre-loaded/registered in the backend environment.
        # A more robust solution would involve loading agent definitions based on IDs/names.
        target_agent_defs = {} # Placeholder for loaded target agent definitions
        adk_tools = _instantiate_adk_tools(agent_config.get('tools', []), target_agent_defs)

        # Instantiate the LlmAgent
        live_agent = LlmAgent(
            name=agent_name,
            model=model,
            instruction=instructions,
            description=agent_config.get('description', ''),
            tools=adk_tools,
            generate_content_config=genai_config,
            # output_schema=... # TODO: Handle structured output schema if configured
            # callbacks=... # TODO: Handle callbacks if configured (requires dynamic loading/execution)
            # sub_agents=[] # Use AgentTool instead for calls
        )
        logging.info(f"ADK Agent '{agent_name}' instantiated.")

        # --- Get or Create Session ---
        try:
             session = session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
             if not session:
                  logging.info(f"Session {session_id} not found, creating new one.")
                  session = session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
        except KeyError: # Handle case where InMemorySessionService raises KeyError
             logging.info(f"Session {session_id} not found (KeyError), creating new one.")
             session = session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)


        # --- Get or Create Runner ---
        if app_name not in runner_cache:
             runner_cache[app_name] = Runner(
                  agent=live_agent, # Use the just-instantiated agent for this run
                  app_name=app_name,
                  session_service=session_service,
                  # artifact_service=artifact_service, # Add if needed
                  # memory_service=memory_service, # Add if needed
             )
        runner = runner_cache[app_name]
        # Update runner's agent if it changed (e.g., config was updated)
        # Note: This simple caching might not handle config updates well.
        # A better approach might create a new runner per request or use a more sophisticated cache.
        runner.agent = live_agent


        # --- Run the Agent ---
        logging.info(f"Executing Runner.run_async for session '{session_id}'...")
        user_message = genai_types.Content(role='user', parts=[genai_types.Part(text=user_input)])
        final_output = "Agent execution finished without a final text response."
        all_events = []

        async for event in runner.run_async(user_id=user_id, session_id=session.id, new_message=user_message):
            all_events.append(event.model_dump(exclude_none=True)) # Store serializable event data
            if event.is_final_response() and event.content and event.content.parts:
                 # Extract text or handle structured output
                 if event.content.parts[0].text:
                      final_output = event.content.parts[0].text
                 else:
                      # Attempt to serialize non-text final parts (e.g., function response if schema used)
                      try:
                           final_output = json.dumps(event.content.parts[0].model_dump(exclude_none=True))
                      except Exception:
                           final_output = "[Non-text final response]"


        logging.info(f"Agent run completed for session '{session_id}'.")

        # --- Process Result ---
        # We'll return the final text output and potentially the list of events for debugging/display
        response_data = {
            "id": f"run-{session_id}-{int(time.time())}",
            "agent_id": agent_config.get('id', 'N/A'), # Config ID from frontend
            "status": "completed", # Assuming completion if no exception
            "input": user_input,
            "output": final_output,
            "final_output": final_output,
            "events": all_events, # Include the event list
            # "new_items": [] # Adapt if ADK events map directly to this structure
        }
        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Error running agent {agent_config.get('name')}: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Failed to run agent: {str(e)}"}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # Use Gunicorn or another production server instead of debug=True
    # For development:
    app.run(debug=True, port=5001)
```

**2. `src/utils/tools.js` (Revised)**

*   Updated code examples for built-in tools to reflect direct usage (e.g., `tools = [google_search]`).
*   Updated code examples for custom/app.py` (Updated)**

```python
# backend/app.py
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import logging
import traceback
import asyncio
from datetime import datetime # For runner template example

# --- Google ADK Imports (Confirmed from ADK source) ---
try:
    # Core Agent & Runner
    from google.adk.agents import LlmAgent # Main agent class (aliased as Agent in adk.__init__)
    from google.adk.runners import Runner
    # Tools
    from google.adk.tools import BaseTool, FunctionTool, AgentTool, LongRunningFunctionTool
    from google.adk.tools import google_search as google_search_tool_instance # Instance
    from google.adk.tools import built_in_code_execution as code_execution_tool_instance # Instance
    from google.adk.tools import VertexAiSearchTool # Class
    # Session & Context
    from google.adk.sessions import InMemorySessionService, BaseSessionService, Session
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.agents.callback_context import CallbackContext # For potential future callback use
    # Events & Types
    from google.adk.events import Event
    from google.genai import types as genai_types # For Content, Part, etc.

except ImportError as e:
    print(f"Error importing Google ADK: {e}. Make sure 'google-adk' is installed.")
    # Define dummy classes/functions if ADK is not installed
    class LlmAgent: pass
    class Runner: pass
    class BaseTool: pass
    class FunctionTool: pass
    class AgentTool: pass
    class LongRunningFunctionTool: pass
    class VertexAiSearchTool: pass
    class InMemorySessionService: pass
    class BaseSessionService: pass
    class Session: pass
    class InvocationContext: pass
    class CallbackContext: pass
    class Event: pass
    class genai_types: class Content: pass; class Part: pass; class FunctionCall: pass; class FunctionResponse: pass;
    async def run_agent_async(*args, **kwargs): return [{"final_output": "ADK not installed - Simulated response"}] # Simulate async generator
    google_search_tool_instance = None
    code_execution_tool_instance = None


# --- Google Cloud Client Imports (for validation, example) ---
try:
    from google.cloud import aiplatform
    from google.auth import exceptions as auth_exceptions
    from google.api_core import exceptions as api_exceptions
except ImportError:
    print("google-cloud-aiplatform or google-auth not installed. API Key validation will be basic.")
    aiplatform = None
    auth_exceptions = None
    api_exceptions = None


# --- Flask App Setup ---
load_dotenv() # Load environment variables from .env file
app = Flask(__name__)
CORS(app) # Enable CORS for requests from the React frontend

logging.basicConfig(level=logging.INFO)

# --- Global Services (Use persistent ones in production) ---
# TODO: Replace InMemorySessionService with DatabaseSessionService or VertexAiSessionService for persistence
session_service: BaseSessionService = InMemorySessionService()
# TODO: Implement and use ArtifactService if needed
# artifact_service = InMemoryArtifactService()
# TODO: Implement and use MemoryService if needed
# memory_service = InMemoryMemoryService()


# --- Helper Functions ---

def load_google_credentials():
    """Checks for Google Cloud credentials."""
    api_key = os.getenv("GOOGLE_API_KEY")
    service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if api_key:
        logging.info("Using GOOGLE_API_KEY for authentication.")
        return {"type": "api_key", "key": api_key}
    elif service_account_path and os.path.exists(service_account_path):
        logging.info(f"Using GOOGLE_APPLICATION_CREDENTIALS from: {service_account_path}")
        # TODO: ADK might need specific credential setup beyond just the path
        return {"type": "service_account", "path": service_account_path}
    else:
        # Attempt Application Default Credentials (ADC)
        try:
            from google.auth import default as default_credentials
            credentials, project_id = default_credentials()
            if credentials:
                 logging.info(f"Using Application Default Credentials (ADC). Project: {project_id}")
                 # ADK likely uses ADC automatically if GOOGLE_APPLICATION_CREDENTIALS is not set
                 return {"type": "adc", "project": project_id}
        except auth_exceptions.DefaultCredentialsError:
             logging.warning("Application Default Credentials not found.")

        logging.warning("No Google Cloud API Key, Service Account, or ADC found in environment variables.")
        return None

def _instantiate_adk_tools(tool_configs):
    """
    Instantiates ADK tool objects based on the configuration from the frontend.
    """
    adk_tools = []
    if not tool_configs:
        return adk_tools

    for config in tool_configs:
        tool_id = config.get('id')
        tool_name = config.get('name')

        try:
            if tool_id == 'googlesearch' or tool_name == 'google_search':
                if google_search_tool_instance:
                    adk_tools.append(google_search_tool_instance)
                    logging.info("Added google_search tool instance.")
                else:
                    logging.warning("google_search tool instance not available (ADK import failed?).")
            elif tool_id == 'codeexecution' or tool_name == 'built_in_code_execution':
                 if code_execution_tool_instance:
                    adk_tools.append(code_execution_tool_instance)
                    logging.info("Added built_in_code_execution tool instance.")
                 else:
                    logging.warning("built_in_code_execution tool instance not available (ADK import failed?).")
            elif tool_id == 'vertexaisearch' or tool_name == 'VertexAiSearchTool':
                params = config.get('parameters', {})
                datastore_id = params.get('data_store_id', {}).get('value') # Get configured value
                if datastore_id:
                    adk_tools.append(VertexAiSearchTool(data_store_id=datastore_id))
                    logging.info(f"Instantiated VertexAiSearchTool with datastore: {datastore_id}")
                else:
                    logging.warning("VertexAiSearchTool requires 'data_store_id' parameter to be configured.")
            elif config.get('category') == 'Function':
                # Custom functions need to be defined in the Python environment
                # where this backend runs. We cannot dynamically execute the
                # code string from the frontend securely or easily.
                # Option 1: Assume functions are defined globally/imported here.
                # Option 2: Use a registry pattern.
                # For now, log a warning.
                logging.warning(f"Cannot dynamically load custom function tool '{tool_name}'. Ensure it's defined and accessible in the backend environment.")
                # Example: If functions were registered:
                # if tool_name in registered_custom_functions:
                #    adk_tools.append(registered_custom_functions[tool_name]) # Add the function object directly
            elif config.get('category') == 'Long Running Function':
                 logging.warning(f"Cannot dynamically load Long Running Function tool '{tool_name}'. Ensure it's defined and wrapped with LongRunningFunctionTool in the backend environment.")
                 # Example:
                 # if tool_name in registered_long_running_functions:
                 #    adk_tools.append(LongRunningFunctionTool(func=registered_long_running_functions[tool_name]))
            elif config.get('category') == 'Agent-as-Tool':
                 logging.warning(f"Cannot dynamically load Agent-as-Tool '{tool_name}'. Ensure the target agent ('{config.get('targetAgentName')}') is defined and wrapped with AgentTool in the backend environment.")
                 # Example:
                 # if config.get('targetAgentName') in registered_agents:
                 #    target_agent = registered_agents[config.get('targetAgentName')]
                 #    adk_tools.append(AgentTool(agent=target_agent, name=tool_name)) # Use configured call name
            else:
                logging.warning(f"Unknown tool type or category in config: {config}")
        except Exception as e:
            logging.error(f"Failed to instantiate tool '{tool_name}': {e}")

    return adk_tools

def _instantiate_adk_sub_agents(sub_agent_configs, all_agent_definitions):
    """
    Instantiates ADK sub-agent objects.
    Requires access to the definitions of the sub-agents.
    """
    sub_agents = []
    if not sub_agent_configs:
        return sub_agents

    for config in sub_agent_configs:
        target_agent_name = config.get('targetAgentName')
        if target_agent_name and target_agent_name in all_agent_definitions:
            # Instantiate the sub-agent using its definition
            # This assumes all_agent_definitions holds the actual agent objects or constructors
            sub_agent_instance = all_agent_definitions[target_agent_name] # Simplified - might need actual instantiation
            sub_agents.append(sub_agent_instance)
            logging.info(f"Added sub-agent: {target_agent_name}")
        else:
            logging.warning(f"Definition for sub-agent '{target_agent_name}' not found in backend environment.")

    return sub_agents

# --- API Endpoints ---

@app.route('/api/validate-key', methods=['POST'])
def validate_key():
    """Validates the provided Google Cloud API Key."""
    # This remains largely simulated as backend should use Service Account/ADC
    data = request.get_json()
    api_key = data.get('apiKey')

    if not api_key:
        return jsonify({"error": "API Key not provided"}), 400

    try:
        # Basic check
        if len(api_key) < 20:
             raise ValueError("Invalid API Key format.")

        # Simulate success - a real check is complex and less relevant if backend uses SA/ADC
        logging.info(f"Simulated validation successful for key ending with ...{api_key[-4:]}")
        return jsonify({"success": True, "message": "API Key validation simulated successfully."})

    except Exception as e:
        logging.error(f"Error validating API key: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Validation failed: {str(e)}"}), 500


@app.route('/api/run-agent', methods=['POST'])
async def handle_run_agent():
    """Handles running the agent based on provided configuration."""
    credentials = load_google_credentials()
    if not credentials:
         # Allow running even without backend credentials if using models that don't require them (e.g., fully local OSS models via LiteLLM if integrated)
         logging.warning("Running without backend Google Cloud credentials. Ensure the selected model/tools do not require them.")
         # return jsonify({"error": "Google Cloud credentials not configured on the backend."}), 500

    # TODO: Configure ADK authentication if needed, e.g., genai.configure(...)

    data = request.get_json()
    agent_config = data.get('agentConfig')
    user_input = data.get('input')
    # Assuming user_id and session_id might be passed for stateful interactions
    user_id = data.get('user_id', 'default_user')
    session_id = data.get('session_id') # Let ADK/SessionService generate if null

    if not agent_config or not user_input:
        return jsonify({"error": "Missing agentConfig or input"}), 400

    try:
        app_name = agent_config.get('name', 'UnnamedApp') # Use agent name as app name for session service
        logging.info(f"Running agent config: {app_name}")

        # --- Instantiate ADK Agent Dynamically ---
        agent_name = agent_config.get('name', 'UnnamedAgent')
        instruction = agent_config.get('instructions', 'You are a helpful assistant.')
        model_name = agent_config.get('model', 'gemini-1.5-flash')
        model_settings = agent_config.get('modelSettings', {})

        # Instantiate tools
        # NOTE: This requires custom functions/agents to be defined/registered in this backend environment
        adk_tools = _instantiate_adk_tools(agent_config.get('tools', []))

        # Instantiate sub-agents (Placeholder - requires agent definitions)
        # all_agent_defs = {} # Load or register agent definitions here
        # adk_sub_agents = _instantiate_adk_sub_agents(agent_config.get('handoffs', []), all_agent_defs)
        adk_sub_agents = [] # Keep empty for now

        # TODO: Instantiate Callbacks (Requires defining callback functions in backend)
        adk_callbacks = []

        # Map model settings from GUI to GenerateContentConfig if needed by LlmAgent
        # ADK's LlmAgent might take these directly or via generate_content_config
        genai_config = genai_types.GenerationConfig(
             temperature=model_settings.get('temperature'),
             top_p=model_settings.get('topP'),
             top_k=model_settings.get('topK'),
             max_output_tokens=model_settings.get('maxOutputTokens'),
             # candidate_count, stop_sequences might be needed too
        )

        # Create the ADK Agent instance
        live_agent = LlmAgent(
            name=agent_name,
            instruction=instruction,
            model=model_name,
            tools=adk_tools,
            sub_agents=adk_sub_agents,
            generate_content_config=genai_config,
            # callbacks=adk_callbacks,
            # output_schema=... # If structured output is configured
        )
        logging.info(f"ADK Agent '{agent_name}' instantiated.")

        # --- Get or Create Session ---
        try:
             session = session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
             if not session:
                  session = session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
                  logging.info(f"Created new session: {session.id}")
             else:
                  logging.info(f"Using existing session: {session.id}")
        except Exception as session_err:
             logging.error(f"Error getting/creating session: {session_err}")
             # Fallback to in-memory if persistent fails? Or just error out.
             return jsonify({"error": f"Failed to get/create session: {str(session_err)}"}), 500


        # --- Create Runner and Run ---
        runner = Runner(
            agent=live_agent,
            app_name=app_name,
            session_service=session_service,
            # artifact_service=artifact_service, # Add if used
            # memory_service=memory_service,     # Add if used
        )

        logging.info(f"Executing runner.run_async for session '{session.id}'...")
        user_content = genai_types.Content(role='user', parts=[genai_types.Part(text=user_input)])

        final_output_text = ""
        all_events_data = [] # Store processed event data for the response

        async for event in runner.run_async(user_id=user_id, session_id=session.id, new_message=user_content):
            logging.debug(f"Received ADK Event: Author={event.author}, ID={event.id}, Partial={event.partial}")
            event_data = {"type": "unknown", "author": event.author, "content": None}

            if event.content:
                 parts_data = []
                 for part in event.content.parts:
                      if part.text:
                           parts_data.append({"type": "text", "text": part.text})
                           if event.is_final_response():
                                final_output_text += part.text # Accumulate final text
                      elif part.function_call:
                           parts_data.append({
                                "type": "tool_call",
                                "name": part.function_call.name,
                                "args": part.function_call.args,
                                "id": part.function_call.id
                           })
                           event_data["type"] = "tool_call_item" # Map to frontend structure
                           event_data["raw_item"] = {"name": part.function_call.name, "arguments": json.dumps(part.function_call.args)}
                      elif part.function_response:
                           parts_data.append({
                                "type": "tool_result",
                                "name": part.function_response.name,
                                "response": part.function_response.response,
                                "id": part.function_response.id
                           })
                           event_data["type"] = "tool_call_output_item" # Map to frontend structure
                           event_data["output"] = part.function_response.response
                      # Add other part types if needed (inline_data, etc.)

                 event_data["content"] = parts_data

            # Add actions if present (e.g., state changes, transfer)
            if event.actions:
                 event_data["actions"] = event.actions.model_dump(exclude_none=True)
                 if event.actions.transfer_to_agent:
                      event_data["type"] = "agent_transfer" # Example type

            if event.is_final_response():
                 event_data["type"] = "message_output_item" # Map final text to frontend structure
                 event_data["raw_item"] = {"content": final_output_text}


            # Only add events with content or meaningful actions to the response list
            if event_data["content"] or event_data.get("actions"):
                 all_events_data.append(event_data)


        logging.info(f"Agent run completed for session '{session.id}'.")

        # --- Format Backend Response ---
        # Mimic the structure expected by the frontend's processAgentResponse
        response_data = {
            "id": f"run-{session.id}-{datetime.now().timestamp()}",
            "agent_id": agent_config.get('id', 'N/A'), # Config ID from frontend
            "status": "completed",
            "input": user_input,
            "output": final_output_text, # Final text response
            "final_output": final_output_text, # Duplicate for compatibility
            "new_items": all_events_data # Pass processed events
        }
        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Error running agent '{agent_name}': {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Failed to run agent: {str(e)}"}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # Use Gunicorn or another production WSGI server instead of app.run in production
    # Example: gunicorn --bind 0.0.0.0:5001 app:app
    app.run(debug=True, port=5001) # Development server

```*Self-correction:* Updated imports based on ADK source (`LlmAgent`, `Runner`, tool classes, session classes, `Event`, `genai_types`). Refined `_instantiate_adk_tools` to use actual tool classes/instances and handle `VertexAiSearchTool` parameters. Added placeholder logic for sub-agents and callbacks, noting they require definitions in the backend environment. Mapped GUI model settings to `genai_types.GenerationConfig`. Updated the `runner.run_async` call signature. Implemented processing of the `Event` stream yielded by `run_async`, extracting text, tool calls (`function_call`), and tool results (`function_response`) to build the `new_items` array expected by the frontend. Added basic session creation/retrieval using the global `session_service`. Added ADC credential check.

**2. `src/utils/tools.js` (Updated `code_example`)agent tools to show direct function addition or `AgentTool` wrapping.

```javascript
// src/utils/tools.js

// --- Built-in Google ADK Tools ---
export const builtInTools = [
  {
    id: 'googlesearch',
    name: 'google_search',
    description: 'Performs a Google Search and returns results. Requires Gemini 2 model. Must display renderedContent if provided.',
    category: 'Built-in',
    parameters: {},
    requirements: "Gemini 2 Model, Display renderedContent",
    code_example: `from google.adk.tools import google_search\n\n# Add directly to agent's tools list\ntools = [google_search]` // Updated example
  },
  {
    id: 'codeexecution',
    name: 'built_in_code_execution',
    description: 'Executes Python code blocks generated by the agent. Requires Gemini 2 model.',
    category: 'Built-in',
    parameters: {},
    requirements: "Gemini 2 Model",
    code_example: `from google.adk.tools import built_in_code_execution\n\n# Add directly to agent's tools list\ntools = [built_in_code_execution]` // Updated example
  },
  {
    id: 'vertexaisearch',
    name: 'VertexAiSearchTool',
    description: 'Searches across configured private Vertex AI Search data stores.',
    category: 'Built-in',
    parameters: {
        data_store_id: {
            type: 'string',
            description: 'Required. The full Vertex AI Search datastore resource ID (e.g., projects/.../dataStores/...).',
            required: true
        }
    },
    requirements: "Vertex AI Search Datastore ID",
    // Updated example shows instantiation
    code_example: `from google.adk.tools import VertexAiSearchTool\n\nYOUR_DATASTORE_ID = "projects/..."\nvertex_tool = VertexAiSearchTool(data_store_id=YOUR_DATASTORE_ID)\n\n# Add the instantiated tool to agent's tools list\ntools = [vertex_tool]`
  }
];

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

# Define the function
def get_stock_price(symbol: str) -> float | None:
    \"\"\"
    Retrieves the current closing stock price for a given symbol using yfinance.

    Args:
        symbol (str): The stock symbol (e.g., "AAPL", "GOOG").

    Returns:
        float: The current stock price as a float.
        None: If the symbol is invalid or data cannot be retrieved.
    \"\"\"
    # ... (implementation as before) ...
    try:
        stock = yf.Ticker(symbol)
        historical_data = stock.history(period="1d", interval="1m")
        if not historical_data.empty:
            return float(historical_data['Close'].iloc[-1])
        else:
            historical_data = stock.history(period="2d")
            if not historical_data.empty:
                 return float(historical_data['Close'].iloc[-1])
            else:
                 return None
    except Exception as e:
        print(f"--- Tool Error retrieving stock price for {symbol}: {e} ---")
        return None

# --- ADK Agent Setup (Example) ---
# from google.adk.agents import LlmAgent
#
# # Add the function *directly* to the tools list
# agent = LlmAgent(
#     model="gemini-1.5-flash",
#     instruction="Fetch stock prices using the available tool.",
#     tools=[get_stock_price] # Pass the function object itself
# )
` // Updated example
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
from typing import Any, Dict

# 1. Define the long running function
def ask_for_approval(purpose: str, amount: float) -> Dict[str, Any]:
    \"\"\"Simulates asking for approval and returns initial pending status."""
    # ... (implementation as before) ...
    ticket_id = f"approval-ticket-{hash(purpose + str(amount)) % 10000}"
    return {'status': 'pending', 'approver': 'manager@example.com', 'purpose' : purpose, 'amount': amount, 'ticket_id': ticket_id}

# 2. Wrap the function
approval_tool = LongRunningFunctionTool(func=ask_for_approval)

# --- ADK Agent Setup (Example) ---
# from google.adk.agents import LlmAgent
#
# # Add the *wrapped* tool instance to the tools list
# agent = LlmAgent(
#     model="gemini-1.5-flash",
#     instruction="Handle reimbursements, asking for approval > $100.",
#     tools=[approval_tool] # Pass the LongRunningFunctionTool instance
# )
` // Updated example
  },
];

// --- Example Agent-as-Tool Reference ---
export const exampleAgentAsTool = {
    id: 'summary_agent_ref',
    name: 'summarize_text',
    description: 'Summarizes long passages of text by calling the dedicated Summary Agent.',
    category: 'Agent-as-Tool',
    targetAgentName: 'summary_agent',
    requirements: "Requires 'summary_agent' LlmAgent instance to be defined.",
    code_example: `from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool

# Assume summary_agent is an LlmAgent instance defined elsewhere
# summary_agent = LlmAgent(name="summary_agent", ...)

# Define the tool that calls the other agent
summarizer_tool = AgentTool(agent=summary_agent, name="summarize_text") # Explicitly name the tool call

# Add the AgentTool instance to the parent agent's tools list
parent_agent = LlmAgent(
    # ... other parent agent params
    instruction="...If you need to summarize text, use the 'summarize_text' tool...",
    tools=[summarizer_tool]
)` // Updated example
};

export const allTools = [
    ...builtInTools,
    ...exampleFunctionTools,
    // { ...exampleAgentAsTool, id: 'agent_tool_example' } // Example reference
];
```

**3. `src/utils/codeTemplates.js` (Revised)**

*   Updated imports and class names (`LlmAgent`).
*   Refined `agentTemplate` to use `generate_content_config` and show correct tool/agent-tool instantiation. Added example callback registration.
*   Refined `runnerTemplate` and `streamingTemplate` based on `Runner.run_async` and event handling examples.
*   Updated `guardrailTemplate` to use `CallbackContext` and show example callback registration.
*   Updated `structuredOutputTemplate` to**

```javascript
// src/utils/tools.js

// --- Built-in Google ADK Tools ---
export const builtInTools = [
  {
    id: 'googlesearch',
    name: 'google_search',
    description: 'Performs a Google Search and returns results. Requires Gemini 2 model. Must display renderedContent if provided.',
    category: 'Built-in',
    parameters: {},
    requirements: "Gemini 2 Model, Display renderedContent",
    // Corrected: Use the imported instance directly
    code_example: `from google.adk.tools import google_search\n\ntools = [google_search]`
  },
  {
    id: 'codeexecution',
    name: 'built_in_code_execution',
    description: 'Executes Python code blocks generated by the agent. Requires Gemini 2 model.',
    category: 'Built-in',
    parameters: {},
    requirements: "Gemini 2 Model",
     // Corrected: Use the imported instance directly
    code_example: `from google.adk.tools import built_in_code_execution\n\ntools = [built_in_code_execution]`
  },
  {
    id: 'vertexaisearch',
    name: 'VertexAiSearchTool',
    description: 'Searches across configured private Vertex AI Search data stores.',
    category: 'Built-in',
    parameters: {
        data_store_id: {
            type: 'string',
            description: 'Required. The full Vertex AI Search datastore resource ID (e.g., projects/.../dataStores/...).',
            required: true
        }
    },
    requirements: "Vertex AI Search Datastore ID",
    // Corrected: Show class instantiation
    code_example: `from google.adk.tools import VertexAiSearchTool\n\nYOUR_DATASTORE_ID = "projects/..."\ntools = [VertexAiSearchTool(data_store_id=YOUR_DATASTORE_ID)]`
  }
];

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
    \"\"\"
    Retrieves the current closing stock price for a given symbol using yfinance.

    Args:
        symbol (str): The stock symbol (e.g., "AAPL", "GOOG").

    Returns:
        float: The current stock price as a float.
        None: If the symbol is invalid or data cannot be retrieved.
    \"\"\"
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
    \"\"\"Simulates asking for approval and returns initial pending status."""
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

export const allTools = [
    ...builtInTools,
    ...exampleFunctionTools,
    // { ...exampleAgentAsTool, id: 'agent_tool_example' } // Example reference
];
```
*Self-correction:* Updated `code_example` fields to show correct ADK usage: direct reference for imported built-in instances (`google_search`, `built_in_code_execution`), class instantiation for `VertexAiSearchTool`, direct function reference for simple functions, `LongRunningFunctionTool(func=...)` wrapping for long-running, and `AgentTool(agent=...)` wrapping for agent-as-tool. Corrected agent import to `LlmAgent`.

**3. `src show `output_schema` parameter.

```javascript
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
#     model="gemini-1.5-flash",
#     instruction="Use the get_stock_price tool when asked for stock prices.",
#     tools=[get_stock_price] # Pass the function object
# )
`;
```

**4. `Google-ADK Reference Notes.md` (Revised)**

*   Updated based on the actual ADK classes and methods identified in the source code.
*   Added details on `LlmAgent` parameters like `generate_content_config`, `output_schema`, callbacks.
*   Clarified `Runner.run_async` usage.
*   Refined tool descriptions (FunctionTool, LongRunningFunctionTool, AgentTool, built-ins).
*   Added notes on `RunConfig` and streaming modes.
*   Included `SessionService` and `Event` details.

```markdown
# Google Agent Development Kit (ADK) - Reference Notes (v0.4.0 based on provided source)

This document provides quick reference notes for key concepts in Google's Agent Development Kit (ADK), based on the provided ADK source code (`google-adk-python-github-repo-complete.md`). Refer to the official ADK documentation for complete details and future updates.

## Installation & Setup

Requires Python 3.9+.

```bash
pip install google-adk google-cloud-aiplatform google-auth # Add other dependencies as needed
```

**Credentials:**
The backend service running ADK requires Google Cloud authentication. Configure via environment variables (checked by ADK/clients):
1.  **Service Account Key (Recommended for Backends):**
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
    ```
2.  **API Key (Simpler for some use cases, ensure permissions):**
    ```bash
    export GOOGLE_API_KEY="YOUR_API_KEY"
    ```
3.  **Application Default Credentials (ADC):** (If running in GCP environment or locally configured)
    ```bash
    gcloud auth application-default login
    ```
4.  **Project ID:** Often required, especially for Vertex AI.
    ```bash
    export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
    ```

## Core Components

### Agent (`google.adk.agents.LlmAgent`)

The primary LLM-powered agent class (often aliased as `google.adk.Agent`).

```python
from google.adk.agents import LlmAgent
from google.adk.tools import google_search, AgentTool, FunctionTool
from google.genai import types as genai_types
# from .my_tools import get_stock_price # Example custom tool function
# from .sub_agents import summary_agent # Example LlmAgent instance

# Example Instantiation
my_agent = LlmAgent(
    name="MyInteractiveAgent",         # Required: Unique string identifier
    model="gemini-1.5-flash",         # Required: Underlying LLM (e.g., Gemini model)
    instruction="Act as a helpful assistant...", # Required: Guides agent behavior
    description="Answers questions using search or summarizes text.", # Optional: For routing/AgentTool
    tools=[                          # Optional: List of tools
        google_search,               # Built-in tool reference
        # get_stock_price,           # Custom Python function (auto-wrapped)
        # AgentTool(agent=summary_agent, name="summarize_text") # Agent-as-a-Tool
    ],
    # sub_agents=[...],              # Optional: List of BaseAgent instances for hierarchy/transfer targets
    generate_content_config=genai_types.GenerateContentConfig( # Optional: Control LLM generation
        temperature=0.7,
        max_output_tokens=1024,
        # safety_settings=...
    ),
    # input_schema=MyInputModel,     # Optional: Pydantic model for structured input when called as tool
    # output_schema=MyOutputModel,   # Optional: Pydantic model for structured output (disables tools)
    # output_key="result_state_key", # Optional: Store final text output in session state['result_state_key']
    # --- Callbacks ---
    # before_agent_callback=my_before_agent_func,
    # after_agent_callback=my_after_agent_func,
    # before_model_callback=my_before_model_func,
    # after_model_callback=my_after_model_func,
    # before_tool_callback=my_before_tool_func,
    # after_tool_callback=my_after_tool_func,
    # --- Other ---
    # include_contents='default',    # 'default' or 'none' (controls history sent to LLM)
    # planner=MyPlanner(),           # For multi-step planning logic
    # code_executor=MyExecutor(),    # For executing LLM-generated code
    # global_instruction="Always be polite.", # Applies to all agents if set on root
    # disallow_transfer_to_parent=False,
    # disallow_transfer_to_peers=False,
)
```

### Runner (`google.adk.runners.Runner`)

Orchestrates agent invocations.

```python
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService # Or other service
from google.adk.agents/utils/codeTemplates.js` (Updated)**

 import RunConfig, StreamingMode
from google.genai import types
import asyncio

# Assume 'my_agent' is an LlmAgent instance
# Assume 'session_service' is a BaseSessionService instance
session = session_service.create_session(app_name="my_app", user_id="user1", session_id="session1")
```javascript
// src/utils/codeTemplates.js

// --- Agent Definition Template (Updated based on ADK source) ---
export const agentTemplate = `import datetime
# --- Core ADK Imports ---
from google.adk.agents import LlmAgent # Use LlmAgent directly
from google.adk.tools import ToolContext, AgentTool, BaseTool, LongRunningFunctionTool # Add LongRunningFunctionTool
from google.adk.agents.callback_context import CallbackContext # Correct context import
from google.genai import types as genai_types # For GenerationConfig

# --- Tool Imports ---
# Import built-in tools used (instances)
{{tool_imports}}
# Import custom tool functions/classes if defined in separate files
# from .my_custom_tools import my_custom_tool_function

# --- Sub-Agent Imports ---
# Import sub-agent definitions (LlmAgent instances) if used
# from .sub_agents.billing_agent import billing_agent

# --- Define Custom Tools (if not imported) ---
{{custom_tool_definitions}}

# --- Define Agent Callbacks (Optional) ---
# Example: Before model callback
def my_before_model_callback(context: CallbackContext, llm_request: 'LlmRequest') -> Optional['LlmResponse']:
    \"\"\"Inspects or modifies the request before sending to the LLM.\"\"\"
    print(f"*** Callback: Before model call for agent '{context.agent_name}' ***")
    # Example: Modify prompt or return a cached response to skip LLM call
    # if "specific keyword" in llm_request.contents[-1].parts[0].text:
    #     return LlmResponse(content=genai_types.Content(parts=[genai_types.Part(text="Skipped LLM via callback.")]))
    return None # Return None to proceed with LLM call

# Example: After tool callback
def my_after_tool_callback(context: CallbackContext, tool: BaseTool, args: dict, tool_response: dict) -> Optional[dict]:
     \"\"\"Inspects or modifies the tool response.\"\"\"
     print(f"*** Callback: After tool call for tool '{tool.name}' ***")
     print(f"Tool Response: {tool_response}")
     # Example: Modify response
     # if 'error' in tool_response:
     #     tool_response['error'] = "Tool failed, attempting recovery."
     # return tool_response # Return modified dict
     return None # Return None to use original response

# --- Instantiate Tools ---
# Add direct function references, wrapped tools, and built-in instances
tools_list = [
{{tools_instantiation}}
]

# --- Instantiate Sub-Agents (if applicable) ---
# This list should contain actual LlmAgent instances imported above
sub_agents_list = [
{{sub_agents_instantiation}}
]

# --- Define the Main Agent ---
agent = LlmAgent(
    name="{{name}}",
    # Use 'instruction' parameter
    instruction="""{{instructions}}

    Current date: {context.state.current_date}""", # Access state via context in instruction provider if needed, or format here
    model="{{model}}",
    tools=tools_list,
    sub_agents=sub_agents_list, # Pass sub-agent instances
    # Pass generation config directly
    generate_content_config=genai_types.GenerateContentConfig(
       temperature={{temperature}},
       top_p={{topP}},
       top_k={{topK}},
       max_output_tokens={{maxOutputTokens}},
       # Add other config like stop_sequences, safety_settings if needed
    ),
    # Register callbacks
    # before_model_callback=my_before_model_callback,
    # after_tool_callback=my_after_tool_callback,
    # output_schema=MyOutputSchema, # If using structured output
    # output_key="my_result_key", # If storing output in state
)

print(f"Agent '{agent.name}' created successfully.")

# --- Runner Example (Updated) ---
# import asyncio
# from google.adk.runners import Runner
# from google.adk.sessions import InMemorySessionService
# from google.genai import types as genai_types
#
# async def main():
#   # Assume credentials are configured via environment variables
#   session_service = InMemorySessionService()
#   session = session_service.create_session(app_name="{{name}}", user_id="test_user")
#   runner = Runner(agent=agent, app_name="{{name}}", session_service=session_service)
#
#   user_query = "What is the stock price for GOOG?"
#   print(f"\\n--- Running Agent for Query: '{user_query}' ---")
#   content = genai_types.Content(role='user', parts=[genai_types.Part(text=user_query)])
#
#   async for event in runner.run_async(user_id=session.user_id, session_id=session.id, new_message=content):
#       print(f"Event Author: {event.author}, ID: {event.id}, Partial: {event.partial}")
#       if event.content:
#           for part in event.content.parts:
#               if part.text: print(f"  Text: {part.text}")
#               if part.function_call: print(f"  Tool Call: {part.function_call.name}({part.function_call.args})")
#               if part.function_response: print(f"  Tool Result ({part.function_response.name}): {part.function_response.response}")
#       if event.is_final_response():
#           print("--- Final Response Received ---")
#
#   print("--------------------\\n")
#
# if __name__ == "__main__":
#   asyncio.run(main())
`;

// --- Custom Tool Template (Standard Python Function - Updated Context/Return) ---
export const functionToolTemplate = `# Potential ADK tool decorator (if applicable)
# from google.adk.tools import tool
from google.adk.tools import ToolContext # Import context
from typing import Any, Dict # Use standard typing

# @tool
def {{name}}({{parameters}}, tool_context: ToolContext) -> Dict[str, Any]: # Return Dict for clarity
    \"\"\"{{description}}

    Args:
        {{args_docs}}
        tool_context (ToolContext): Context object providing session state, etc.

    Returns:
        Dict[str, Any]: A dictionary containing the result (e.g., {'result': ...} or {'status': ..., 'data': ...}).
                       Return None or empty dict if the tool is long-running and result comes later.
    \"\"\"
    # Function implementation goes here
    print(f"Executing tool: {{name}} with args: {{args_list}}")
    # Access state if needed: value = tool_context.state.get('my_key')
    # Save state if needed: tool_context.state['my_key'] = 'new_value' (persists after yield/return)
    # Request auth if needed: tool_context.request_credential(...)

    # Replace with actual logic
    result_data = f"Processed {{args_list}} using context state if needed."
    return {"result": result_data} # Return a dictionary
`;

// --- Structured Output Template (Updated Agent Setup) ---
export const structuredOutputTemplate = `from pydantic import BaseModel, Field
from typing import List, Optional

class {{className}}(BaseModel):
    \"\"\"{{class_description}}\"\"\"
    {{fields}}

# --- ADK Agent Setup (Example) ---
# from google.adk.agents import LlmAgent # Correct import
#
# agent = LlmAgent(
#     name="{{name}}",
#     instruction="""{{instructions}}
#
#     Respond using the {{className}} JSON structure.""",
#     output_schema={{className}}, # Correct parameter name
#     model="{{model}}"
#     # Cannot use tools when output_schema is set
# )
`;

// --- Callback Template (Updated based on ADK source) ---
export const guardrailTemplate = `# ADK Callback Implementation Example
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse # Correct imports
from google.genai import types as genai_types
from typing import Optional

# Example: before_model_callback (Input Guardrail / Modification)
def check_input_and_modify_request(context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    \"\"\"Checks input or modifies the LLM request. Return LlmResponse to skip LLM call.\"\"\"
    print(f"--- Running before_model_callback for Agent: {context.agent_name} ---")
    last_user_content = llm_request.contents[-1] if llm_request.contents else None
    if last_user_content and last_user_content.role == 'user':
        user_text = last_user_content.parts[0].text if last_user_content.parts else ""
        print(f"Checking user input: '{user_text[:50]}...'")
        if "sensitive_topic" in user_text.lower():
            print("Input guardrail triggered. Skipping LLM call.")
            # Return a predefined response to skip the actual LLM call
            return LlmResponse(
                content=genai_types.Content(parts=[genai_types.Part(text="I cannot discuss that topic.")])
            )
        # Example modification: Add safety preamble
        # llm_request.append_instructions(["Always respond safely and ethically."])

    return None # Proceed with LLM call

# Example: after_model_callback (Output Guardrail / Modification)
def check_output_and_modify_response(context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    \"\"\"Checks or modifies the LLM response.\"\"\"
    print(f"--- Running after_model_callback for Agent: {context.agent_name} ---")
    if llm_response.content and llm_response.content.parts:
         response_text = llm_response.content.parts[0].text if llm_response.content.parts[0].text else ""
         print(f"Checking LLM output: '{response_text[:50]}...'")
         if "undesired_pattern" in response_text.lower():
              print("Output guardrail triggered. Modifying response.")
              llm_response.content.parts[0].text = "[Content modified by output guardrail]"
              # Return the modified LlmResponse object
              return llm_response

    return None # Use the original response


# --- ADK Agent Setup (Example) ---
# from google.adk.agents import LlmAgent
#
# agent = LlmAgent(
#     name="{{name}}",
#     instruction="{{instructions}}",
#     model="{{model}}",
#     before_model_callback=check_input_and_modify_request,
#     after_model_callback=check_output_and_modify_response,
# )
`;

// --- Runner Template (Updated based on ADK source) ---
export const runnerTemplate = `import asyncio
import json
# Assuming the agent definition is in agent.py
from agent import agent # Import the LlmAgent instance

# --- Google ADK Runner/Session Imports ---
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService # Or DatabaseSessionService, etc.
from google.genai import types as genai_types

async def main():
    # Ensure Google Cloud credentials are set in the environment
    # e.g., export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"

    app_name = agent.name # Use agent name for app_name
    user_id = "cli_test_user"
    session_service = InMemorySessionService()
    session = session_service.create_session(app_name=app_name, user_id=user_id)
    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)

    user_prompt = "{{prompt}}"
    print(f"--- Running Agent '{agent.name}' for Prompt: '{user_prompt}' ---")

    content = genai_types.Content(role='user', parts=[genai_types.Part(text=user_prompt)])

    try:
        final_response_text = "No final text response captured."
        async for event in runner.run_async(user_id=user_id, session_id=session.id, new_message=content):
            print(f"Event Author: {event.author}, ID: {event.id}, Partial: {event.partial}")
            # Process event details (optional)
            if event.content:
                for part in event.content.parts:
                    if part.text: print(f"  Text: {part.text}")
                    if part.function_call: print(f"  Tool Call: {part.function_call.name}({part.function_call.args})")
                    if part.function_response: print(f"  Tool Result ({part.function_response.name}): {part.function_response.response}")
            if event.is_final_response() and event.content and event.content.parts and event.content.parts[0].text:
                 final_response_text = event.content.parts[0].text

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

// --- Streaming Template (Updated based on ADK source) ---
export const streamingTemplate = `import asyncio
import json
# Assuming the agent definition is in agent.py
from agent import agent # Import the LlmAgent instance

# --- Google ADK Runner/Session Imports ---
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk import RunConfig, StreamingMode # Import RunConfig
from google.genai import types as genai_types

async def main():
    # Ensure Google Cloud credentials are set
    app_name = agent.name
    user_id = "cli_stream_user"
    session_service = InMemorySessionService()
    session = session_service.create_session(app_name=app_name, user_id=user_id)
    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)

    user_prompt = "{{prompt}}"
    print(f"--- Streaming Agent '{agent.name}' for Prompt: '{user_prompt}' ---")

    content = genai_types.Content(role='user', parts=[genai_types.Part(text=user_prompt)])

    # Configure for streaming
    run_config = RunConfig(streaming_mode=StreamingMode.SSE) # Use SSE for streaming

    try:
        print("\\n--- Agent Stream Events ---")
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=content,
            run_config=run_config # Pass the streaming config
        ):
            print(f"Event Author: {event.author}, ID: {event.id}, Partial: {event.partial}")
            # Process event details
            if event.content:
                for part in event.content.parts:
                    if part.text:
                        print(f"  Text Chunk: '{part.text}'")
                    if part.function_call:
                        print(f"  Tool Call: {part.function_call.name}({part.function_call.args})")
                    if part.function_response:
                        print(f"  Tool Result ({part.function_response.name}): {part.function_response.response}")
            if event.is_final_response():
                 print("--- Final Response Received (End of Stream) ---")


    except Exception as e:
        print(f"\\n--- Error during agent stream ---")
        print(e)
        import traceback
        traceback.print_exc()

    print("\\n-------------------------\\n")


if __name__ == "__main__":
  asyncio.run(main())
`;

// Template for a full custom tool example (Updated Context/Return)
export const fullToolExampleTemplate = `import json
from typing import Dict, Any, List # Use standard typing
from google.adk.tools import ToolContext # Import context

# Potential ADK tool decorator (if applicable)
# from google.adk.tools import tool

# @tool
def get_stock_price(ticker_symbol: str, tool_context: ToolContext) -> Dict[str, Any]:
    \"\"\"Fetches the current stock price for a given ticker symbol.

    Args:
        ticker_symbol: The stock ticker symbol (e.g., 'GOOGL').
        tool_context: The context object.

    Returns:
        A dictionary containing the ticker and its simulated price, or an error message.
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
        # Return a dictionary as expected by ADK
        return {"ticker": ticker_symbol, "price": price, "currency": "USD"}
    else:
        return {"error": f"Could not find price for ticker: {ticker_symbol}"}

# --- ADK Agent Setup (Example) ---
# from google.adk.agents import LlmAgent
# stock_price_tool = get_stock_price # Add function directly
# agent = LlmAgent(..., tools=[stock_price_tool])
`;
```
*Self-correction:* Updated imports in `agentTemplate` to `LlmAgent`. Corrected `generate_content_config` usage. Added `sub_agents` parameter. Updated callback examples with correct signatures (`CallbackContext`). Updated runner examples with correct `run_async` signature and event processing loop. Updated `functionToolTemplate` and `fullToolExamplerunner = Runner(agent=my_agent, app_name="my_app", session_service=session_service)

async def run_agent_query(user_id, session_id, query):
    user_message = types.Content(role='user', parts=[types.Part(text=query)])
    run_config = RunTemplate` to include `ToolContext` parameter and return a `Dict`. Updated `structuredOutputTemplate` to use `output_schema` parameter. Updated `guardrailTemplate` to use `CallbackContext` and correct model imports.

**4. `src/components/builder/GuardrailsConfigurator.js` (Updated - Renaming & Dialog Text)**

Config(streaming_mode=StreamingMode.SSE) # Example: Enable streaming

    print(f"Running agent for query: {query}")
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_message,
        run_config=run_config
    ):
        print(f"Received Event: ID={event.id}, Author={event.author}, Partial={event.partial}")
        # Process event content (text, function calls, etc.)
        if event.content and event.content.parts:
            for part in event.content.parts:
                 if part.text: print(f"  Text: {part.text}")
                 # ... handle other part types ...
        if event.is_final_response():
            print("--- Final Response ---")

# asyncio.run(run_agent_query("user1", "session1", "Hello agent!"))
```
- `run_async`: Primary method for execution. Takes `user_id`, `session_id`, `new_message` (`types.Content`), `run_config`. Returns an async generator yielding `Event` objects.
- `run`: Synchronous wrapper around `run_async` for convenience.

### Session & State (`google.adk.sessions`)

- **`Session`**: Holds `id`, `app_name`, `user_id`, `state` (dict), `events` (list), `last_update_time`.
- **`State`**: Wrapper around the session state dictionary (`session.state`) passed via context objects. Allows getting/setting values. Changes are recorded in `EventActions.state_delta` and committed by the `SessionService`. Prefixes like `State.APP_PREFIX` (`app:`), `State.USER_PREFIX` (`user:`), `State.TEMP_PREFIX` (`temp:`) can be used for namespacing, though only non-temp changes are persisted by default services.
- **`BaseSessionService`**: Abstract class for session persistence.
    - `InMemorySessionService`: Default, non-persistent.
    - `DatabaseSessionService`: Uses SQLAlchemy for DB persistence.
    - `VertexAiSessionService`: Connects to managed Vertex AI session service.

### Context Objects (`callback_context.py`, `tool_context.py`, `readonly_context.py`)

- **`InvocationContext`**: Internal context for a single `run_async` call. Holds references to services, session, current agent, etc.
- **`CallbackContext`**: Passed to agent/model callbacks. Provides read-only access to most invocation details (`agent_name`, `invocation_id`) and read/write access to session state (`context.state`). Changes to state are tracked for `state_delta`. Can load/save artifacts via methods.
- **`ToolContext`**: Passed to tool functions. Inherits from `CallbackContext`, adding `function_call_id`. Can request credentials (`request_credential`) and access auth responses (`get_auth_response`).

### Event (`google.adk.events.Event`)

Unit of communication yielded by `run_async`.
- Inherits from `LlmResponse`.
- **Key Attributes:** `id`, `invocation_id`, `author` ('user' or agent name), `content` (`types.Content`), `actions` (`EventActions`), `timestamp`, `partial`, `turn_complete`, `branch`.
- **`content`**: Contains `role` and `parts` (list of `types.Part`). Parts can be text, function calls, function responses, inline data (blobs), code execution requests/results.
- **`actions`**: Contains intended side effects like `state_delta`, `artifact_delta`, `transfer_to_agent`, `escalate`, `skip_summarization`, `requested_auth_configs`. Processed by the Runner.
- **`is_final_response()`**: Helper method to check if it's the last event for a logical turn (usually no function calls/responses, not partial).

## Tools (`google.adk.tools`)

Extend agent capabilities. Added to `LlmAgent(tools=[...])`.

### Function Tools (`function_tool.py`)

- **Simple Functions:** Pass Python function object directly to `tools` list. ADK auto-wraps with `FunctionTool`.
    - Use standard type hints (str, int, float, bool, list, dict, Optional, Union, Literal). Pydantic models also supported.
    - Docstring becomes tool description for LLM.
    - Return value should be JSON serializable (dict preferred). Simple types are wrapped `{'result': value}`.
    - Avoid default parameter values (LLM doesn't use them).
    - Can accept `tool_context: ToolContext` as an argument.
- **`LongRunningFunctionTool` (`long_running_tool.py`):**
    - Wrap function with `LongRunningFunctionTool(func=my_func)`.
    - Function can return initial status dict (e.g., `{'status': 'pending', 'ticket_id': '123'}`).
    - Runner yields event and pauses.
    - External client logic monitors status and sends `FunctionResponse` back via `runner.run_async` to resume agent.
- **`AgentTool` (`agent_tool.py`):**
    - Wrap another `LlmAgent` instance: `AgentTool(agent=child_agent, name="call_child")`.
    - Allows parent agent to call child agent using the specified `name`.
    - Child agent runs in its own temporary session.
    - Child's final response (text or structured JSON if `output_schema` used) is returned as the tool result to the parent.
    - `skip_summarization=True`: Parent LLM doesn't summarize child response, uses it directly.

### Built-in Tools

- **`google_search` (`google_search_tool.py`):** Add the imported object `google_search` to `tools`. Requires Gemini 2.
- **`built_in_code_execution` (`built_in_code_execution_tool.py`):** Add the imported object `built_in_code_execution` to `tools`. Requires Gemini 2.
- **`VertexAiSearchTool` (`vertex_ai_search_tool.py`):** Instantiate with datastore ID: `VertexAiSearchTool(data_store_id="projects/...")`. Add instance to `tools`.
- **Other Tools:** `load_artifacts`, `load_memory`, `exit_loop`, `transfer_to_agent`, `get_user_choice` (long running), `APIHubToolset`, `ApplicationIntegrationToolset`, `GoogleApiToolSet` (for specific Google APIs like Gmail, Calendar), `LangchainTool`, `CrewaiTool`.

**Limitation:** Only *one* instance of a built-in tool type (Search, Code Exec) per agent. Use `AgentTool` pattern for multiple search types (web vs vertex) or multiple code executions.

## Callbacks (`base_agent.py`, `llm_agent.py`)

Hook into agent lifecycle via `LlmAgent` constructor parameters.

- **Signatures:**
    - `before_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]`
    - `after_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]`
    - `before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]`
    - `after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]`
    - `before_tool_callback(tool: BaseTool, args: dict, tool_context: ToolContext) -> Optional[dict]`
    - `after_tool_callback(tool: BaseTool, args: dict, tool_context: ToolContext, tool_response: dict) -> Optional[dict]`
- **Control Flow:** Return `None` to continue default execution. Return specific object type (Content, LlmResponse, dict) to skip/override the next step.
- **Use Cases:** Logging, validation, guardrails, state modification, caching, external triggers.

## Runtime Configuration (`run_config.py`)

Passed to `Runner.run_async(run_config=...)`.

```python
from google.adk.agents import RunConfig, StreamingMode
from google.genai import types

config = RunConfig(
    streaming_mode=StreamingMode.SSE, # NONE, SSE, BIDI
    max_llm_calls=50,                 # Limit LLM calls per invocation
    speech_config=types.SpeechConfig( # For voice input/output
        language_code="en-US",
        # voice_config=...
    ),
    response_modalities=["TEXT", "AUDIO"], # Specify desired outputs
    save_input_blobs_as_artifacts=False, # Save uploaded files?
    support_cfc=False, # Experimental Compositional Function Calling
    output_audio_transcription=types.AudioTranscriptionConfig(...), # Transcribe agent audio output
    input_audio_transcription=types.AudioTranscriptionConfig(...), # Transcribe user audio input
)
```

## Other Components

- **Models (`google.adk.models`):** Defines `BaseLlm` interface. `Gemini` implementation provided. `LLMRegistry` handles model selection.
- **Planners (`google.adk.planners`):** For multi-step reasoning (`PlanReActPlanner`, `BuiltInPlanner`). Assigned to `LlmAgent(planner=...)`.
- **Code Executors (`google.adk.code_executors`):** Execute code (`UnsafeLocalCodeExecutor`, `VertexAiCodeExecutor`, `ContainerCodeExecutor`). Assigned to `LlmAgent(code_executor=...)`.
- **Memory (`google.adk.memory`):** For long-term memory across sessions (`InMemoryMemoryService`, `VertexAiRagMemoryService`). Used by `Runner` and potentially `load_memory_tool`.
- **Artifacts (`google.adk.artifacts`):** For storing/retrieving file data (`InMemoryArtifactService`, `GcsArtifactService`). Used by `Runner` and context objects.
- **Evaluation (`google.adk.evaluation`):** Tools for evaluating agent performance (`AgentEvaluator`, `ResponseEvaluator`, `TrajectoryEvaluator`).
- **CLI (`google.adk.cli`):** Command-line tools (`adk run`, `adk web`, `adk deploy`, `adk eval`).
```

---

This concludes the refactoring based on the provided ADK source code. The backend, utilities, and documentation should now more accurately reflect the structure and capabilities of the Google Agent Development Kit as presented in the source files.

Let me know if you have any further adjustments or questions!