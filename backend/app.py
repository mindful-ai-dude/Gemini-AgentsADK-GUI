# backend/app.py
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import logging
import traceback

# --- Google ADK Imports ---
# Adjust these imports based on the actual ADK structure
try:
    from google.adk import Agent as GoogleAgent
    from google.adk import run_agent_async # Assuming an async runner function exists
    from google.adk.tools import ToolContext, AgentTool, BaseTool # Assuming BaseTool exists
    # Import specific built-in tools if needed by name
    from google.adk.tools import google_search # Example based on docs
    # Import context/state management if needed
    # from google.adk.core import State, InvocationContext
except ImportError as e:
    print(f"Error importing Google ADK: {e}. Make sure it's installed.")
    # Define dummy classes/functions if ADK is not installed, for basic structure
    class GoogleAgent: pass
    class ToolContext: pass
    class AgentTool: pass
    class BaseTool: pass
    def run_agent_async(*args, **kwargs): return {"final_output": "ADK not installed - Simulated response"}
    def google_search(*args, **kwargs): return "Simulated Google Search result"


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

# --- Helper Functions ---

def load_google_credentials():
    """Checks for Google Cloud credentials."""
    api_key = os.getenv("GOOGLE_API_KEY")
    service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if api_key:
        logging.info("Using GOOGLE_API_KEY for authentication.")
        # Note: Many Google Cloud client libraries prefer service accounts for backends.
        # API Key usage might be limited or require specific setup per library.
        return {"type": "api_key", "key": api_key}
    elif service_account_path and os.path.exists(service_account_path):
        logging.info(f"Using GOOGLE_APPLICATION_CREDENTIALS from: {service_account_path}")
        return {"type": "service_account", "path": service_account_path}
    else:
        logging.warning("No Google Cloud API Key or Service Account found in environment variables.")
        return None

def _instantiate_adk_tools(tool_configs):
    """
    Instantiates ADK tool objects based on the configuration from the frontend.
    This needs refinement based on actual ADK tool definition and registration.
    """
    adk_tools = []
    if not tool_configs:
        return adk_tools

    for config in tool_configs:
        tool_id = config.get('id')
        tool_name = config.get('name')
        # Basic mapping - needs to be more robust based on ADK specifics
        if tool_id == 'googlesearch' or tool_name == 'google_search':
             # Assuming google_search is a class or function ready to be used
             try:
                 # ADK might require specific instantiation or registration
                 # This is a placeholder assumption
                 adk_tools.append(google_search())
                 logging.info(f"Instantiated Google Search tool.")
             except Exception as e:
                 logging.error(f"Failed to instantiate google_search: {e}")
        # TODO: Add logic for other built-in tools (File Search, etc.)
        # elif tool_id == 'filesearch':
        #     adk_tools.append(FileSearchTool(**config.get('parameters', {})))
        elif config.get('category') == 'Function':
            # Custom function tools are defined in Python code.
            # The backend would need access to this code or a way to dynamically load/define it.
            # This is complex and likely requires a more advanced setup.
            # For now, we'll log a warning.
            logging.warning(f"Custom function tool '{tool_name}' cannot be dynamically instantiated in this basic backend setup.")
            # Placeholder: Create a dummy tool
            # class DummyTool(BaseTool): name=tool_name; description=config.get('description'); async def __call__(self, *args, **kwargs): return f"Simulated output for {self.name}"
            # adk_tools.append(DummyTool())
        else:
            logging.warning(f"Unknown tool type or category: {config}")

    return adk_tools

def _instantiate_adk_sub_agents(sub_agent_configs):
    """
    Instantiates ADK sub-agent objects (representing handoffs).
    This is highly dependent on how ADK handles sub-agents/handoffs.
    """
    sub_agents = []
    if not sub_agent_configs:
        return sub_agents

    for config in sub_agent_configs:
        # ADK likely requires defining these sub-agents similar to the root agent.
        # This implies the backend needs the definitions for these agents too.
        # Complex to handle dynamically without a predefined agent registry.
        logging.warning(f"Sub-agent/Handoff '{config.get('name')}' cannot be dynamically instantiated in this basic setup.")
        # Placeholder:
        # sub_agent = GoogleAgent(name=config.get('name'), ...) # Requires full definition
        # sub_agents.append(sub_agent)

    return sub_agents


# --- API Endpoints ---

@app.route('/api/validate-key', methods=['POST'])
def validate_key():
    """Validates the provided Google Cloud API Key."""
    data = request.get_json()
    api_key = data.get('apiKey')

    if not api_key:
        return jsonify({"error": "API Key not provided"}), 400

    # --- Actual Validation Logic ---
    # Replace this with a robust check using a Google Cloud client library
    try:
        if not aiplatform:
             # Basic check if client library isn't available
             if len(api_key) > 20: # Arbitrary length check
                 logging.info("Basic API Key format check passed (google-cloud-aiplatform not installed).")
                 # IMPORTANT: This doesn't actually validate the key with Google!
                 return jsonify({"success": True, "message": "API Key format looks basic, but not fully validated."})
             else:
                 raise ValueError("Invalid API Key format.")

        # Example using Vertex AI client library (adjust project/location if needed)
        project_id = os.getenv("GOOGLE_PROJECT_ID")
        if not project_id:
             logging.warning("GOOGLE_PROJECT_ID not set, validation might be limited.")
             # Fallback to basic check or skip validation if project ID is essential

        # Use API Key for client initialization (may require specific library setup)
        # Note: Service accounts are generally preferred for backend auth.
        # This example assumes API key auth is feasible for the chosen validation call.
        # You might need to adjust authentication based on the library used.
        # client_options = {"api_key": api_key} if api_key else None # Example structure
        # aiplatform.init(project=project_id, location='us-central1', client_options=client_options) # Example init

        # Perform a simple, low-cost operation like listing models
        # This specific call might change depending on library versions and auth method
        # await aiplatform.ModelServiceClient().list_models(parent=f"projects/{project_id}/locations/us-central1") # Example async call

        # Simulate success for now as direct API key validation with client libraries can be tricky
        logging.info(f"Simulated validation successful for key ending with ...{api_key[-4:]}")
        return jsonify({"success": True, "message": "API Key validation simulated successfully."})

    except auth_exceptions.GoogleAuthError as e:
        logging.error(f"Authentication error during validation: {e}")
        return jsonify({"error": f"Authentication failed: {e}"}), 401
    except api_exceptions.PermissionDenied as e:
         logging.error(f"Permission denied during validation: {e}")
         return jsonify({"error": "Permission denied. Check API key permissions."}), 403
    except Exception as e:
        logging.error(f"Error validating API key: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Validation failed: {str(e)}"}), 500


@app.route('/api/run-agent', methods=['POST'])
async def handle_run_agent():
    """Handles running the agent based on provided configuration."""
    credentials = load_google_credentials()
    if not credentials:
         return jsonify({"error": "Google Cloud credentials not configured on the backend."}), 500

    # TODO: Set up ADK authentication using credentials if required by the library
    # Example: google_adk.set_api_key(credentials['key']) or configure service account auth

    data = request.get_json()
    agent_config = data.get('agentConfig')
    user_input = data.get('input')

    if not agent_config or not user_input:
        return jsonify({"error": "Missing agentConfig or input"}), 400

    try:
        logging.info(f"Running agent: {agent_config.get('name')}")

        # --- Instantiate ADK Agent Dynamically ---
        # This requires careful mapping from the GUI config to ADK Agent parameters
        agent_name = agent_config.get('name', 'Unnamed Agent')
        instructions = agent_config.get('instructions', 'You are a helpful assistant.')
        model = agent_config.get('model', 'gemini-1.5-flash') # Default Google model
        # TODO: Map modelSettings (temperature, topP, topK, maxOutputTokens) to ADK equivalents
        model_settings = agent_config.get('modelSettings', {})

        # Instantiate tools and sub-agents (placeholders for now)
        adk_tools = _instantiate_adk_tools(agent_config.get('tools', []))
        adk_sub_agents = _instantiate_adk_sub_agents(agent_config.get('handoffs', [])) # Map handoffs

        # TODO: Instantiate Guardrails based on config

        # Create the ADK Agent instance
        # The exact parameters will depend on the google-adk Agent class definition
        live_agent = GoogleAgent(
            name=agent_name,
            instructions=instructions,
            model=model,
            # tools=adk_tools, # Pass instantiated tools
            # sub_agents=adk_sub_agents, # Pass instantiated sub-agents
            # model_config=model_settings, # Pass model settings if applicable
            # input_guardrails=...,
            # output_guardrails=...,
        )
        logging.info(f"ADK Agent '{agent_name}' instantiated (structure based on assumptions).")

        # --- Run the Agent ---
        # Assuming an async run function like `run_agent_async(agent, user_input)`
        # The actual function and parameters might differ in the real ADK.
        # It might require a context object as well.
        logging.info(f"Executing run_agent_async for '{agent_name}'...")
        result = await run_agent_async(live_agent, user_input) # Await the async function
        logging.info(f"Agent run completed. Result type: {type(result)}")

        # --- Process Result ---
        # Adapt this based on the actual structure returned by the ADK run function
        if isinstance(result, dict):
            # Assuming result is a dictionary similar to the simulation structure
            final_output = result.get("final_output", "No output found in result.")
            new_items = result.get("new_items", []) # Pass through any trace items if available
            run_id = result.get("id", f"run-{agent_config.get('id')}-{Date.now()}")
            status = result.get("status", "completed")
        elif isinstance(result, str):
             # If the run function just returns a string
             final_output = result
             new_items = []
             run_id = f"run-{agent_config.get('id')}-{Date.now()}"
             status = "completed"
        else:
             # Handle unexpected result types
             logging.error(f"Unexpected result type from ADK run: {type(result)}")
             final_output = f"Unexpected result type: {type(result)}"
             new_items = []
             run_id = f"run-{agent_config.get('id')}-{Date.now()}"
             status = "error"


        response_data = {
            "id": run_id,
            "agent_id": agent_config.get('id'),
            "status": status,
            "input": user_input,
            "output": final_output, # Use the processed output
            "final_output": final_output,
            "new_items": new_items # Include trace items if the ADK provides them
        }
        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Error running agent {agent_config.get('name')}: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Failed to run agent: {str(e)}"}), 500

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True, port=5001) # Run on a different port than the React app (e.g., 5001)