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
_ADK_FULL_IMPORT_SUCCESS = True
AgentTool = None # Initialize AgentTool

try:
    # Core Agent & Runner
    from google.adk.agents import LlmAgent # Main agent class (aliased as Agent in adk.__init__)
    from google.adk.runners import Runner
    # Tools - Import AgentTool separately
    from google.adk.tools import BaseTool, FunctionTool, LongRunningFunctionTool
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

    try:
        from google.adk.tools import AgentTool as RealAgentTool
        AgentTool = RealAgentTool
        logging.info("Successfully imported AgentTool from google.adk.tools.")
    except ImportError:
        logging.warning("`AgentTool` could not be imported from `google.adk.tools`. Defining a dummy class. Agent-as-Tool functionality will be limited.")
        class _DummyAgentTool: # Define dummy AgentTool specifically here
            def __init__(self, *args, **kwargs):
                logging.warning("Dummy AgentTool instantiated.")
                pass
        AgentTool = _DummyAgentTool
        globals()['AgentTool'] = _DummyAgentTool # Ensure it's globally available

except ImportError as e:
    _ADK_FULL_IMPORT_SUCCESS = False
    print(f"Error importing critical Google ADK components: {e}. Make sure 'google-adk' is installed. Defining dummy classes for all ADK components.")
    # Define dummy classes/functions if ADK is not installed or core components fail
    class LlmAgent: pass
    class Runner: pass
    class BaseTool: pass
    class FunctionTool: pass
    if AgentTool is None: # Define AgentTool dummy only if not defined by the inner try-except
        class _FallbackDummyAgentTool: pass
        AgentTool = _FallbackDummyAgentTool
        globals()['AgentTool'] = _FallbackDummyAgentTool
    class LongRunningFunctionTool: pass
    class VertexAiSearchTool: pass
    class InMemorySessionService: pass
    class BaseSessionService: pass
    class Session: pass
    class InvocationContext: pass
    class CallbackContext: pass
    class Event: pass
    class genai_types:
        class Content: pass
        class Part: pass
        class FunctionCall: pass
        class FunctionResponse: pass
    async def run_agent_async(*args, **kwargs): return [{"final_output": "ADK not installed - Simulated response"}]
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
        model_name = agent_config.get('model', 'gemini-2.5-pro-preview-05-06')
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