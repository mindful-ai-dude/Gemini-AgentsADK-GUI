**1. New File: `Google-ADK Reference Notes.md`**

```markdown
# Google Agent Development Kit (ADK) - Reference Notes

This document provides quick reference notes for key concepts in Google's Agent Development Kit (ADK), based on available documentation and examples. Refer to the official ADK documentation for complete details.

## Installation & Setup

Requires Python 3.9+.

```bash
pip install google-adk google-cloud-aiplatform google-auth # Add other dependencies as needed
```

**Credentials:**
The backend service running ADK typically requires Google Cloud authentication. Configure via:
1.  **Service Account Key (Recommended for Backends):**
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
    ```
2.  **API Key (Simpler for some use cases, ensure permissions):**
    ```bash
    export GOOGLE_API_KEY="YOUR_API_KEY"
    ```
3.  **Application Default Credentials (ADC):**
    ```bash
    gcloud auth application-default login
    ```

## Core Components

### Agent (`google.adk.Agent` / `LlmAgent`)

The primary building block, representing an entity that can reason, plan, and act. `LlmAgent` specifically uses a Large Language Model.

```python
from google.adk.agents import LlmAgent # Often aliased as Agent
from google.adk.tools import google_search, AgentTool
# from .my_tools import my_custom_function_tool
# from .sub_agents import other_agent

agent = LlmAgent(
    name="MyAgent",                   # Required: Unique string identifier
    model="gemini-1.5-flash",         # Required: Underlying LLM (e.g., Gemini model)
    instruction="Your prompt here...", # Required: Guides agent behavior, persona, tool use
    description="Optional: Describes agent capabilities for routing.",
    tools=[                          # Optional: List of tools the agent can use
        google_search,               # Built-in tool reference
        # my_custom_function_tool,   # Custom Python function (auto-wrapped)
        # AgentTool(agent=other_agent, name="call_other_agent") # Agent-as-a-Tool
    ],
    # sub_agents=[...],              # Optional: For agent transfer/delegation (ADK specific mechanism)
    # generate_content_config=...,   # Optional: Control LLM generation (temperature, etc.)
    # input_schema=MyInputModel,     # Optional: Pydantic model for structured input
    # output_schema=MyOutputModel,   # Optional: Pydantic model for structured output (disables tools)
    # output_key="result_state_key", # Optional: Store final text output in session state
    # callbacks=[...],               # Optional: Hook into execution lifecycle
    # include_contents='default',    # Optional: Control history ('default' or 'none')
    # planner=MyPlanner(),           # Optional: For multi-step planning
    # code_executor=MyExecutor(),    # Optional: For executing generated code
)
```

### Runner (`google.adk.runners.Runner`)

Orchestrates a single agent invocation (user query processing). Manages the event loop, interacts with services, and yields events.

```python
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio

# Assume 'agent' and 'session_service' are defined
# Assume 'session' is created: session = session_service.create_session(...)
runner = Runner(agent=agent, app_name="my_app", session_service=session_service)

async def run_query(user_id, session_id, query):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_response = "No response."
    # Use run_async for the core asynchronous execution
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        print(f"Event: {event.type}, Author: {event.author}") # Example logging
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts.text
    return final_response

# Example usage
# result = asyncio.run(run_query(USER_ID, SESSION_ID, "What is the capital of Canada?"))
# print(f"Final Result: {result}")
```

### Session & State (`google.adk.sessions`)

- **Session:** Container for a single conversation's state and event history. Managed by `SessionService`.
- **State (`context.state`):** A dictionary within the session accessible via context objects (e.g., `CallbackContext`, `ToolContext`). Used to store and share data across turns or between agent components within an invocation. Changes are committed via `state_delta` in `EventActions`.

### Context Objects (`CallbackContext`, `ToolContext`)

Passed to callbacks and potentially tools. Provide access to invocation details, agent info, session state (`context.state`), and services (like artifact saving).

### Event (`google.adk.events.Event`)

The unit of communication between the Runner and Execution Logic. Contains `author`, `content` (using `google.genai.types`), `actions` (like `state_delta`, `artifact_delta`), `type`, `invocation_id`, etc. Yielded by agents/tools/flows, processed by the Runner.

## Tools (`google.adk.tools`)

Extend agent capabilities.

### Function Tools

- **Simple Functions:** Add Python functions directly to `tools` list. ADK wraps them. Use type hints and clear docstrings. Return type should ideally be `dict` for clarity, otherwise ADK wraps in `{'result': ...}`. Avoid default parameter values.
- **Long Running Function Tools:** Wrap a function with `LongRunningFunctionTool` for tasks that take time. The function can yield intermediate status updates (dicts) and return a final result. The Runner pauses/resumes based on these. Requires client logic to handle updates.
- **Agent-as-a-Tool:** Wrap another `Agent` instance with `AgentTool`. Allows the parent agent to call the child agent like a function. The child's response is returned to the parent. Use `name` parameter in `AgentTool` to define the call name. Use `skip_summarization=True` to pass child response directly.

### Built-in Tools

- **`google_search`**: Performs Google Search. Requires Gemini 2. Must display `renderedContent` if provided.
- **`built_in_code_execution`**: Executes Python code generated by the agent. Requires Gemini 2.
- **`VertexAiSearchTool`**: Searches configured Vertex AI Search datastores. Requires `data_store_id` during instantiation.

**Limitation:** Currently, only *one* built-in tool instance per root/single agent. Cannot be used in sub-agents directly (use Agent-as-Tool pattern).

## Callbacks

Python functions registered with an agent to hook into its lifecycle.

- **Registration:** Pass functions to `LlmAgent` constructor (e.g., `before_model_callback=my_func`).
- **Context:** Receive `CallbackContext` providing access to state, agent info, etc.
- **Control Flow:**
    - `return None`: Allow default ADK behavior to proceed.
    - `return <Specific Object>`: Override default behavior (skip LLM call, replace tool result, etc.). See docs for specific return types per callback.
- **Use Cases:** Logging, debugging, input/output validation (guardrails), state management, caching, triggering external actions.

**Available Callbacks (Examples):**
- `before_agent_callback` / `after_agent_callback`
- `before_model_callback` / `after_model_callback`
- `before_tool_callback` / `after_tool_callback`

## Runtime Configuration (`google.adk.RunConfig`)

Passed to `Runner.run` or `run_async` to customize execution.

```python
from google.adk import RunConfig, StreamingMode
from google.genai import types

config = RunConfig(
    streaming_mode=StreamingMode.SSE, # Or NONE, BIDI
    max_llm_calls=100,
    # speech_config=types.SpeechConfig(...), # For voice
    # response_modalities=["TEXT", "AUDIO"],
    # save_input_blobs_as_artifacts=True,
    # support_cfc=True, # Experimental, requires SSE
    # output_audio_transcription=types.AudioTranscriptionConfig(...),
)
```

## Streaming

- Enabled via `RunConfig(streaming_mode=...)`.
- `SSE`: Server-Sent Events (server->client).
- `BIDI`: Bidirectional.
- Runner yields multiple `Event` objects, many with `partial=True`. Final event is non-partial. State changes usually commit only on the final event.

---
*Self-correction:* Added sections based on the provided `google-sdk-docs.md`. Clarified Function Tool parameters/return types/docstrings. Detailed Long Running Function Tools and Agent-as-a-Tool concepts, including the key difference from sub-agents. Listed built-in tools with requirements. Explained Callbacks and the control flow mechanism (return None vs. return Object). Added RunConfig details. Included basic code examples adapted from the docs.

**2. New File: `test-google-api.bat`**

```batch
@echo off
echo Testing Google Cloud API connection via Node.js script...
echo.
echo This script attempts to use the backend's configured credentials
echo (ideally GOOGLE_APPLICATION_CREDENTIALS or gcloud ADC)
echo to perform a simple Google Cloud API action (e.g., list Vertex AI models).
echo Ensure the backend's environment is correctly set up.
echo.
echo Make sure Node.js is installed and required packages are available for the script.
echo Running node src/test-google-api.js...
echo.

node src/test-google-api.js

echo.
echo Test script finished. Check output above for success or errors.
echo If errors occur, verify backend credentials, permissions, and required APIs enabled in Google Cloud.
echo Press any key to exit.
pause > nul
```

**3. New File: `src/test-google-api.js`**
*(Note: This requires Node.js Google Cloud client libraries. It simulates checking credentials accessible to a backend environment)*

```javascript
// src/test-google-api.js
// This script attempts to validate Google Cloud credentials accessible
// to the environment, similar to how the backend would.
// Requires Node.js and relevant Google Cloud libraries installed globally or locally.
// Run `npm install google-auth-library @google-cloud/aiplatform` if needed for this script.

const { GoogleAuth } = require('google-auth-library');
const { PredictionServiceClient } = require('@google-cloud/aiplatform').v1; // Example client

// --- Configuration ---
// Try to get project ID from environment, fallback to a placeholder
const PROJECT_ID = process.env.GOOGLE_PROJECT_ID || 'your-gcp-project-id';
// Use a common region, adjust if needed
const LOCATION = process.env.GOOGLE_CLOUD_LOCATION || 'us-central1';

async function testGoogleCloudApi() {
  console.log(`Attempting to authenticate and connect to Google Cloud API...`);
  console.log(`Using Project ID: ${PROJECT_ID}, Location: ${LOCATION}`);
  console.log(`Credentials should be configured via GOOGLE_APPLICATION_CREDENTIALS env var or gcloud auth application-default login.`);

  let auth;
  try {
    // Attempt to get credentials using Application Default Credentials (ADC)
    auth = new GoogleAuth({
      scopes: ['https://www.googleapis.com/auth/cloud-platform'] // Common scope
    });
    const client = await auth.getClient();
    console.log("Successfully obtained Google Cloud credentials.");

    // --- Perform a simple API call ---
    // Example: List endpoints in Vertex AI (adjust API call as needed)
    // This requires the AI Platform Prediction API to be enabled.
    console.log(`Attempting to list Vertex AI Endpoints in ${LOCATION}...`);
    const aiplatformClient = new PredictionServiceClient({
        authClient: client,
        apiEndpoint: `${LOCATION}-aiplatform.googleapis.com` // Construct regional endpoint
     });


    const parent = `projects/${PROJECT_ID}/locations/${LOCATION}`;
    // Use listEndpoints with callback or promise style depending on library version
     try {
        const [endpoints] = await aiplatformClient.listEndpoints({parent: parent, pageSize: 1}); // Request only 1 to minimize output/cost
        console.log(`Successfully listed ${endpoints.length} Vertex AI endpoint(s) (or 0 if none exist).`);
        console.log("Google Cloud API connection appears to be working.");
        return true;
     } catch (apiError) {
         console.error(`API call failed: ${apiError.message}`);
         if (apiError.code === 7) { // Permission Denied
             console.error("Error: Permission denied. Ensure the credentials have the 'aiplatform.endpoints.list' permission.");
         } else if (apiError.code === 5) { // Not Found (often project ID issue)
              console.error("Error: Project or location might be incorrect, or API not enabled.");
         } else if (apiError.code === 3) { // Invalid Argument
             console.error("Error: Invalid argument, check project ID/location format.");
         }
         // Add more specific error checks based on Google API error codes
         return false;
     }

  } catch (error) {
    console.error('Error during Google Cloud API test:', error.message);
    if (error.message.includes('Could not load the default credentials')) {
        console.error("Hint: Run 'gcloud auth application-default login' or set GOOGLE_APPLICATION_CREDENTIALS.");
    }
    return false;
  }
}

// Run the test
testGoogleCloudApi().then(success => {
  if (success) {
    console.log('\nSUCCESS: Google Cloud API connection test passed!');
  } else {
    console.log('\nFAILED: Google Cloud API connection test failed. Check credentials, permissions, project ID, and ensure required APIs are enabled.');
  }
}).catch(err => {
    console.error("\nUNEXPECTED SCRIPT ERROR:", err);
});
```
*Self-correction:* Renamed files. Updated `.bat` script text. The `.js` script now uses `google-auth-library` and `@google-cloud/aiplatform` (example client) to attempt authentication via ADC and make a simple API call (listing Vertex AI endpoints) to verify connectivity and basic permissions, providing more specific error hints.

**4. `docs/index.md` (Modified)**

```markdown
# Gemini Agent Builder GUI Documentation

Welcome to the documentation for the Gemini Agent Builder GUI. This documentation will help you get started with configuring, testing, and deploying AI agents using **Google's Agent Development Kit (ADK)** via an intuitive graphical interface and a Flask backend.

**IMPORTANT**: This project is created by J. Gravelle (https://j.gravelle.us | j@gravelle.us) and is **not affiliated with, endorsed by, or sponsored by Google**. It is an independent tool designed to facilitate working with the Google Agent Development Kit (ADK).

## Table of Contents

### Getting Started
- [Installation Guide](./getting-started.md#installation) (Includes Frontend & Backend Setup)
- [Setting Up Your Google Cloud Credentials](./getting-started.md#setting-up-your-google-cloud-credentials)
- [Creating Your First Agent Configuration](./getting-started.md#creating-your-first-agent-configuration)
- [Testing Your Agent Configuration](./getting-started.md#testing-your-agent-configuration)
- [Troubleshooting](./getting-started.md#troubleshooting)

### Core Concepts (Google ADK)
- Understanding Agents (`LlmAgent`)
- Working with Tools (Built-in, Function Tools, Agent-as-Tool)
- Callbacks (Lifecycle Hooks)
- Runtime & Execution Flow
- Session Management & State
*(Note: Link these to relevant sections in `Google-ADK Reference Notes.md` or official ADK docs)*

### Advanced Usage
- Custom Function Tools (Python Definition)
- Structured Output (Pydantic Schemas)
- Multi-Agent Patterns (Agent-as-Tool, Sub-Agents if supported)
- Deployment Strategies (Frontend & Flask Backend)
*(Note: Link these to relevant sections or create dedicated pages)*

### Reference
- [Google ADK Reference Notes](./Google-ADK%20Reference%20Notes.md) (Quick Reference)
- [Code Generation](./code-generation.md) (Explains generated Python code)
- [Official Google ADK Documentation](https://google.github.io/adk-docs/) (External Link)

### Contributing
- [Development Setup](./CONTRIBUTING.md)
- Code Style Guide (To be defined)
- Testing (To be defined)
- Documentation (Contribution guidelines)

## About This Documentation

This documentation aims to guide users of the Gemini Agent Builder GUI. For in-depth ADK details, please refer to the official Google ADK documentation. If you find issues or have suggestions, please open an issue on our GitHub repository.

## Additional Resources

- [GitHub Repository](https://github.com/yourusername/gemini-agentsdk-gui-app) <!-- UPDATE LINK -->
- [Google AI for Developers](https://ai.google.dev/)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)

## License

Gemini Agent Builder GUI is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.
```
*Self-correction:* Updated title and description. Changed links and section titles to reflect Google ADK concepts and the new architecture (Agent Configuration, Callbacks, Backend). Added link to the new reference notes file.

**5. `docs/getting-started.md` (Modified)**

```markdown
# Getting Started with Gemini Agent Builder GUI

This guide helps you set up and start using the Gemini Agent Builder GUI and its backend to configure, test, and deploy AI agents using Google's Agent Development Kit (ADK).

**IMPORTANT**: This project is created by J. Gravelle (https://j.gravelle.us | j@gravelle.us) and is **not affiliated with, endorsed by, or sponsored by Google**.

## Prerequisites

Before you begin, ensure you have:

- Node.js (v16.x+) & npm (v8.x+) for the frontend GUI.
- Python (v3.9+) & pip for the backend service.
- Git for cloning the repository.
- **Google Cloud Credentials**:
    - **API Key:** Obtainable from the Google Cloud Console. Ensure it has permissions for required services (e.g., Vertex AI API). **Note:** While the GUI asks for this key for initial validation, the backend service should ideally run with a more secure Service Account.
    - **(Recommended for Backend)** **Service Account Key:** Create a service account in your Google Cloud project, grant it necessary roles (e.g., Vertex AI User), and download its JSON key file.
- A Google Cloud Project with necessary APIs enabled (e.g., Vertex AI API).

## Installation

The application consists of a React frontend and a Flask backend. Both need to be set up.

**1. Clone the Repository**
```bash
git clone https://github.com/yourusername/gemini-agentsdk-gui-app.git # Use the correct repo name
cd gemini-agentsdk-gui-app
```

**2. Set up Backend**
```bash
cd backend
# Create & activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure Credentials (using .env file)
cp .env.example .env
nano .env # Or use your preferred editor

# Inside .env, uncomment and set ONE of the following:
# GOOGLE_API_KEY=AIzaSy..........
# OR (Recommended)
# GOOGLE_APPLICATION_CREDENTIALS=/full/path/to/your/service-account-key.json

# Also set your project ID:
# GOOGLE_PROJECT_ID=your-gcp-project-id

cd .. # Back to root directory
```

**3. Set up Frontend**
```bash
# In the root directory
npm install
```

**4. Run the Application**

*   **Terminal 1: Start Backend**
    ```bash
    cd backend
    source venv/bin/activate # If needed
    flask run --port 5001
    ```
    *(Keep running)*

*   **Terminal 2: Start Frontend**
    ```bash
    # In the root directory
    npm start
    ```

**5. Access the GUI**
Open `http://localhost:3000` in your browser.

## Setting Up Your Google Cloud Credentials (GUI Prompt)

1.  On first launch, the GUI prompts for a Google Cloud API Key.
2.  Enter a valid API key obtained from the Google Cloud Console. This key is primarily used for an **initial validation check** via the backend.
3.  The key is stored locally in your browser *only* to remember that validation passed.
4.  **Crucially, the backend service uses the credentials configured in its `.env` file (ideally a service account) for actual interactions with Google AI services during agent runs.** Ensure the backend credentials have the necessary permissions.

## Creating Your First Agent Configuration

Agent configurations define how an ADK agent should behave.

### Step 1: Basic Details
1.  From the dashboard, click "Create Agent Config".
2.  Enter a **Name** and **Description** for your configuration.
3.  Select a **Gemini Model** (e.g., `gemini-1.5-flash`).
4.  Adjust **Model Settings** (Temperature, Top K, Top P, Max Output Tokens) as needed.
5.  Click "Next".

### Step 2: Instructions
1.  Write the **System Prompt** in the editor.
2.  Clearly define the agent's role, persona, task, constraints, and how/when to use specific tools or sub-agents. Use Markdown for clarity.
3.  You can use templates as a starting point.
4.  Click "Next".

### Step 3: Tools
1.  Add **Built-in Tools** like `google_search` or `built_in_code_execution`. Note the limitation of one built-in tool per agent. Configure required parameters (e.g., Datastore ID for `VertexAiSearchTool`).
2.  Add **Custom Function Tools** by defining Python functions (the GUI helps structure this, but the code runs on the backend).
3.  Click "Next".

### Step 4: Sub-Agents (Agent-as-Tool)
1.  Configure references to other agent configurations that this agent can call.
2.  Define the **Tool Name** used for invocation and provide a description for the parent agent.
3.  This enables multi-agent orchestration patterns.
4.  Click "Next".

### Step 5: Guardrails (Callbacks)
1.  Configure **Callbacks** to hook into the agent's execution lifecycle (e.g., `before_model_callback`, `after_tool_callback`).
2.  Define custom Python functions (executed on the backend) for logging, validation, or modifying behavior. (UI provides structure, implementation is backend).
3.  Configure **Structured Output** using a Pydantic schema if needed (disables tool use for that agent).
4.  Click "Next".

### Step 6: Code Preview
1.  Review the generated Python code (`agent.py`, runner examples) based on your configuration using Google ADK syntax.
2.  Copy or download the code for use in your own ADK projects.
3.  Click "Next".

### Step 7: Test Agent Config
1.  Interact with your agent configuration via the chat interface.
2.  Messages are sent to the Flask backend, which instantiates and runs the ADK agent.
3.  Observe responses and check backend logs for execution details.
4.  Go back through the steps to refine the configuration as needed.
5.  Click "Save Config" (or "Update Config") when finished.

## Next Steps

- Explore the [Google ADK Reference Notes](./Google-ADK%20Reference%20Notes.md).
- Consult the [Official Google ADK Documentation](https://google.github.io/adk-docs/).
- Check out our [Example Agent Configurations](./example-agents.md).
- Learn about [Deployment Options](./deployment.md) for the frontend and backend.

## Troubleshooting

If you encounter issues:

1.  **Backend Not Running:** Ensure the Flask server (`python backend/app.py` or `flask run --port 5001`) is running in its terminal.
2.  **Credential Errors:** Verify backend credentials (`.env` file) are correct and have necessary Google Cloud permissions (e.g., Vertex AI User role). Check if required APIs are enabled in your GCP project.
3.  **API Key Validation Failure (GUI):** Ensure the API key entered in the GUI is valid and hasn't expired. Check backend logs for specific validation errors.
4.  **Agent Run Errors:** Check the backend terminal logs for detailed error messages from the ADK or Google APIs. Ensure instructions correctly guide tool/sub-agent usage.
5.  **UI Issues:** Clear your browser cache or try a different supported browser.
6.  **ADK Limitations:** Remember the current limitation regarding only one built-in tool per agent.

## Getting Help

- Check this guide and the [FAQ](./faq.md).
- Review the [Google ADK Reference Notes](./Google-ADK%20Reference%20Notes.md).
- Consult the [Official Google ADK Documentation](https://google.github.io/adk-docs/).
- Open an issue on the project's GitHub repository for bugs or feature requests.
```
*Self-correction:* Updated prerequisites, installation, credential setup, and workflow steps to match the Google ADK + Flask backend architecture. Clarified the role of the API key entered in the GUI vs. backend credentials. Renamed steps/concepts (e.g., "Agent Configuration", "Sub-Agents", "Callbacks"). Updated troubleshooting tips.

**6. `docs/deployment.md` (Modified)**

```markdown
# Deployment Guide (Gemini Agent Builder GUI & Backend)

This guide provides instructions for deploying the Gemini Agent Builder GUI (React frontend) and its associated Flask backend service.

**IMPORTANT**: This project is created by J. Gravelle (https://j.gravelle.us | j@gravelle.us) and is **not affiliated with, endorsed by, or sponsored by Google**.

## Deploying the Frontend (React GUI)

The frontend is a standard React application built into static files.

1.  **Build the Application:**
    ```bash
    # From the project root directory
    npm run build
    ```
    This creates optimized static files (HTML, CSS, JS) in the `build/` directory.

2.  **Host the Static Files:**
    Deploy the contents of the `build/` directory using any static hosting service:
    *   **Google Cloud Storage + Load Balancer:** Host files in a GCS bucket and serve via Cloud Load Balancing for HTTPS and CDN.
    *   **Firebase Hosting:** Easy deployment integrated with Google Cloud.
    *   **Netlify / Vercel:** Popular platforms for static site deployment.
    *   **Self-Hosted Web Server:** Use Nginx, Caddy, or Apache to serve the static files. Ensure your server configuration handles single-page application routing (redirecting all paths to `index.html`).

    **Example Nginx Configuration Snippet:**
    ```nginx
    server {
        listen 80;
        server_name your-agent-builder-domain.com;

        root /path/to/your/frontend/build; # Path to the build directory
        index index.html;

        location / {
            try_files $uri $uri/ /index.html; # Handle SPA routing
        }

        # Add SSL configuration (e.g., Let's Encrypt) for HTTPS
        # listen 443 ssl;
        # ssl_certificate /path/to/cert.pem;
        # ssl_certificate_key /path/to/key.pem;
        # ... other SSL settings ...
    }
    ```

## Deploying the Backend (Flask Service)

The backend runs the Google ADK agents and requires a Python environment. **Do not run the Flask development server (`flask run`) in production.**

1.  **Prepare the Backend Code:** Ensure the `backend/` directory contains `app.py`, `requirements.txt`, and any necessary helper modules.

2.  **Configure Credentials:** Set environment variables securely in your deployment environment. **Do not hardcode credentials.**
    *   `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json` (Recommended)
    *   `GOOGLE_API_KEY=AIza...` (If using API key, ensure security)
    *   `GOOGLE_PROJECT_ID=your-gcp-project-id`

3.  **Choose a Deployment Method:**
    *   **Google Cloud Run:** A fully managed serverless platform. Containerize the Flask app.
        *   Create a `backend/Dockerfile`.
        *   Build and push the container image to Google Artifact Registry.
        *   Deploy to Cloud Run, configuring environment variables for credentials.
    *   **Google App Engine (Standard or Flex):** Platform-as-a-Service. Configure `app.yaml`.
    *   **Google Compute Engine (GCE) / Other VM:** Deploy manually on a virtual machine.
        *   Set up a Python environment.
        *   Install dependencies (`pip install -r requirements.txt`).
        *   Use a production WSGI server like Gunicorn or Waitress.
        *   Use a process manager (like `systemd` or `supervisor`) to keep the service running.
        *   (Recommended) Set up a reverse proxy (Nginx, Caddy) in front of the WSGI server to handle requests, SSL termination, etc.

    **Example using Gunicorn (on a VM/Container):**
    ```bash
    # Assuming you are in the backend directory with venv activated
    # Install Gunicorn in your virtual environment
    pip install gunicorn

    # Run the Flask app with Gunicorn (adjust workers as needed)
    gunicorn --workers 4 --bind 0.0.0.0:8080 app:app
    ```
    *(You would typically run this via a process manager like systemd)*

    **Example `backend/Dockerfile` for Cloud Run:**
    ```dockerfile
    # Use an official Python runtime as a parent image
    FROM python:3.11-slim

    # Set the working directory in the container
    WORKDIR /app

    # Copy the requirements file into the container
    COPY requirements.txt .

    # Install any needed packages specified in requirements.txt
    # Use --no-cache-dir to reduce image size
    RUN pip install --no-cache-dir -r requirements.txt

    # Copy the backend application code into the container
    COPY . .

    # Make port 8080 available to the world outside this container
    # Cloud Run uses this port by default
    EXPOSE 8080

    # Define environment variable for the port (optional, good practice)
    ENV PORT=8080

    # Run app.py when the container launches using Gunicorn
    # Use the production server, not flask run
    CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
    ```

4.  **Configure CORS:** Ensure the Flask backend's CORS configuration (`CORS(app)` in `app.py`) allows requests from your deployed frontend domain. For production, specify origins explicitly: `CORS(app, origins=["https://your-agent-builder-domain.com"])`.

5.  **Update Frontend API URL:** If the backend is deployed to a different domain than the frontend, update the `BACKEND_URL` constant in `src/utils/apiService.js` to point to the deployed backend URL.

## Production Considerations

*   **Credentials Security:** Use Service Accounts for the backend whenever possible. Manage keys securely (e.g., Secret Manager). Avoid committing keys to version control.
*   **Scalability:** Cloud Run and App Engine offer automatic scaling. For VMs, configure load balancing and auto-scaling groups if needed.
*   **Monitoring & Logging:** Use Google Cloud Logging and Monitoring (or other tools) to track backend performance, errors, and API usage.
*   **Error Handling:** Implement robust error handling in the Flask backend to catch ADK errors, API errors, and provide informative responses to the frontend.
*   **Rate Limiting:** Be aware of Google Cloud API rate limits. Implement retries with backoff in the backend if necessary.
*   **HTTPS:** Ensure both frontend and backend are served over HTTPS. Use Google Managed SSL certificates or configure your own.
*   **Dependencies:** Keep both frontend (npm) and backend (pip) dependencies updated.
```
*Self-correction:* Separated deployment instructions for frontend (static files) and backend (Flask/Python). Added specific examples for different hosting options (GCS, Firebase, Nginx for frontend; Cloud Run, App Engine, GCE/Gunicorn for backend). Included an example Dockerfile for the backend. Emphasized using a production WSGI server (Gunicorn) instead of `flask run`. Added notes on CORS configuration and updating the frontend's `BACKEND_URL`. Reinforced security best practices for credentials.

**7. `docs/example-agents.md` (Modified)**

```markdown
# Example Agent Configurations (Google ADK)

This document provides examples of agent *configurations* you can create with the Gemini Agent Builder GUI, designed to work with the Google Agent Development Kit (ADK) via the backend service. Use these as inspiration.

**IMPORTANT**: This project is created by J. Gravelle (https://j.gravelle.us | j@gravelle.us) and is **not affiliated with, endorsed by, or sponsored by Google**.

## 1. Google Cloud Customer Support Router

### Description
A first-level support agent that understands user intent regarding Google Cloud, answers basic questions, uses Google Search for current info, and routes complex queries to specialized agent tools (Database Agent, BQML Agent).

### Configuration (in GUI)

**Basic Details:**
- Name: `GCP Support Router`
- Model: `gemini-1.5-flash` (or `gemini-1.5-pro` for better routing)
- Description: Routes Google Cloud inquiries to appropriate specialists or answers basic questions using search.

**Instructions:**
```markdown
You are a helpful Google Cloud Platform support router. Your primary goal is to understand the user's intent and either answer their question directly using Google Search or route them to the correct specialist tool.

**Workflow:**
1.  **Understand Intent:** Determine if the user is asking a general question, needs to query data (SQL), or wants to perform a BigQuery ML task.
2.  **General Questions:** If it's a general question about GCP services, pricing, or recent news, use the `google_search` tool to find the latest information and provide a summarized answer. Cite your sources if possible.
3.  **Data Query (SQL):** If the user asks to query data, analyze tables, or asks a question requiring specific data from BigQuery (e.g., "show me sales data for Canada"), use the `call_db_agent` tool. Pass the user's natural language query directly to this tool.
4.  **BigQuery ML:** If the user specifically asks about training a model, forecasting, or other BQML tasks, use the `call_bqml_agent` tool. Pass the user's request to this tool.
5.  **Unclear/Complex:** If the intent is unclear, ask clarifying questions. If the query is too complex or outside your scope (e.g., detailed security architecture), state that you cannot help and suggest consulting the official GCP documentation or support.

**Constraints:**
- Prioritize using the `call_db_agent` or `call_bqml_agent` tools if the query clearly matches their purpose.
- Only use `google_search` for general information retrieval.
- Be polite and professional.
```

**Tools:**
- `google_search` (Built-in)
- `call_db_agent` (Agent-as-Tool reference, configured in Sub-Agents tab)
- `call_bqml_agent` (Agent-as-Tool reference, configured in Sub-Agents tab)

**Sub-Agents (Agent-as-Tool References):**
- Configure references pointing to the `Database Query Agent (NL2SQL)` and `BQML Forecasting Agent` configurations (assuming they exist).

**Guardrails (Callbacks):**
- (Optional) Could add a `before_model_callback` to log the routing decision.

## 2. Research Assistant (using Google Search)

### Description
An agent focused on web research using the built-in Google Search tool, summarizing findings, and citing sources.

### Configuration (in GUI)

**Basic Details:**
- Name: `Web Research Assistant`
- Model: `gemini-1.5-pro` (Good for summarization)
- Description: Performs web searches using Google Search and summarizes results.

**Instructions:**
```markdown
You are a research assistant. Your primary capability is using Google Search to find information on topics provided by the user.

**Process:**
1.  Receive the user's research topic or question.
2.  Formulate effective search queries based on the user's request.
3.  Use the `google_search` tool to execute the queries.
4.  Analyze the search results provided by the tool.
5.  Synthesize the information from the relevant search results into a concise summary.
6.  Present the summary to the user.
7.  **Crucially:** If the `google_search` tool provides rendered content (citations/links), you MUST include this information clearly in your final response to the user, as per policy requirements. Mention the sources used.
8.  If the search tool fails or returns no relevant results, inform the user you couldn't find information on that topic.

**Tone:** Factual, objective, and informative.
```

**Tools:**
- `google_search` (Built-in)

**Sub-Agents (Agent-as-Tool References):**
- None needed for this simple example.

**Guardrails (Callbacks):**
- (Optional) An `after_model_callback` could be used to double-check that source attribution/renderedContent (if provided by the tool event) is included in the final text response.

## 3. Simple Finance Calculator (Custom Function Tool)

### Description
An agent that uses a custom Python function (defined via the GUI, executed by the backend) to perform basic calculations.

### Configuration (in GUI)

**Basic Details:**
- Name: `Simple Calculator Agent`
- Model: `gemini-1.5-flash`
- Description: Performs basic arithmetic using a custom tool.

**Instructions:**
```markdown
You are a simple calculator. When the user provides a basic arithmetic expression (addition, subtraction, multiplication, division), use the `simple_calculator` tool to compute the result.

**Example:**
User: Calculate 15 * 4
You: *Use the `simple_calculator` tool with expression="15 * 4"*
You: The result is 60.

If the user asks for something other than a simple calculation, state that you can only perform basic arithmetic.
```

**Tools:**
- `simple_calculator` (Custom Function Tool, defined in the Tools tab)
    - *Name:* `simple_calculator`
    - *Description:* `Perform a basic arithmetic calculation (+, -, *, /).`
    - *Parameters:* `expression: str` (Description: The mathematical expression to evaluate)
    - *Code:* (Provide the Python code for the `simple_calculator` function as shown in `src/utils/tools.js`)

**Sub-Agents (Agent-as-Tool References):**
- None

**Guardrails (Callbacks):**
- (Optional) A `before_tool_callback` could validate the `expression` parameter to ensure it only contains allowed characters/operators before execution.

## Implementing These Examples

1.  Launch the Gemini Agent Builder GUI (ensure the Flask backend is running).
2.  Navigate to the Dashboard and click "Create Agent Config".
3.  Follow the wizard steps, entering the details provided above for each example.
    *   For the **Tools** step, add the specified built-in or custom function tools. For custom tools, paste the Python code into the editor provided in the GUI.
    *   For the **Sub-Agents** step, configure the references to other agent configs if required by the example.
    *   For **Guardrails (Callbacks)**, the UI might be basic; the core logic needs backend implementation, but you can note the intent here.
4.  Use the **Test Agent Config** tab to interact with your configuration via the backend.
5.  Review the generated Python code in the **Code Preview** tab.
```
*Self-correction:* Updated examples to use Google ADK concepts (Gemini models, `google_search`, Agent-as-Tool, Callbacks). Renamed examples slightly for clarity (GCP Support Router). Made instructions more specific about using Google tools and routing to other agent tools. Clarified that custom function code runs on the backend. Updated the "Implementing" section.

**8. `docs/faq.md` (Modified)**

```markdown
# Frequently Asked Questions (FAQ) - Gemini Agent Builder GUI

## General Questions

### What is the Gemini Agent Builder GUI?
It's a visual interface (React frontend) paired with a Python backend (Flask) for **configuring**, testing, and managing AI agents built using **Google's Agent Development Kit (ADK)**. It allows you to define agent behavior, tools, and interactions without writing all the ADK boilerplate code from scratch, generating usable Python ADK code based on your configuration.

### Is this an official Google tool?
**No.** This is a third-party tool built to work *with* the Google Agent Development Kit (ADK). It is not officially affiliated with, endorsed by, or sponsored by Google. This project is created and maintained by J. Gravelle (https://j.gravelle.us | j@gravelle.us).

### Is the Gemini Agent Builder GUI free to use?
The GUI application itself is open-source (MIT License) and free to use. However, running the agents via the backend service consumes Google Cloud resources (like Vertex AI API calls for Gemini models, potentially BigQuery, Vertex AI Search, etc.), which incur costs based on your Google Cloud usage and pricing.

## API Keys, Credentials, and Pricing

### Where do I get Google Cloud Credentials?
- **API Keys:** Can be generated in the [Google Cloud Console APIs & Services > Credentials](https://console.cloud.google.com/apis/credentials) page. Needed for initial GUI validation.
- **Service Account Keys (Recommended for Backend):** Create a service account in your GCP project, assign appropriate roles (e.g., Vertex AI User, BigQuery User), and download the JSON key file. Configure the backend (`backend/.env`) to use this key file path via `GOOGLE_APPLICATION_CREDENTIALS`.

### How much does it cost to run the agents?
Costs depend on:
- The specific Gemini model used (`gemini-1.5-flash` is generally cheaper than `gemini-1.5-pro`).
- The number of tokens processed in requests and responses.
- Usage of other Google Cloud services invoked by tools (e.g., BigQuery queries, Vertex AI Search calls).
Refer to the [Google Cloud Pricing Calculator](https://cloud.google.com/products/calculator) and specific service pricing pages (like [Vertex AI Pricing](https://cloud.google.com/vertex-ai/pricing)).

### Is my API Key / Service Account Key secure?
- **Frontend (API Key):** The API key entered in the GUI is stored *only* in your browser's local storage and is used just for the initial validation call to the backend. It's not used for running agents.
- **Backend (Credentials):** The backend service uses credentials (ideally a service account key) configured via environment variables (`.env` file or system environment). **Protect your service account key file.** Do not commit it to version control. Follow standard security practices for managing backend credentials.

### Can I use this without Google Cloud credentials?
No. The backend service requires valid Google Cloud credentials (API Key or Service Account) to interact with Google AI services (like Gemini models) via the ADK. The frontend also requires an API key for the initial validation step.

## Features and Functionality

### What models can I use?
You can configure the agent to use Gemini models supported by the Google ADK (e.g., `gemini-1.5-flash`, `gemini-1.5-pro`). The list in the GUI can be updated in `src/utils/modelOptions.js`.

### What tools can I configure?
- **Built-in ADK Tools:** Google Search (`google_search`), Code Execution (`built_in_code_execution`), Vertex AI Search (`VertexAiSearchTool`). Note current limitations (e.g., one built-in per agent).
- **Custom Function Tools:** Define standard Python functions in the GUI; the backend executes them when called by the agent.
- **Agent-as-Tool:** Configure one agent to call another defined agent configuration.

### Can I export my agent configurations?
Yes, the "Code Preview" tab generates Python code using Google ADK based on your GUI configuration. You can copy or download this code (`agent.py`, runner examples) to integrate into your own ADK projects.

### Can I import agent configurations?
Currently, the application does not support importing configurations from external files. This could be a future enhancement.

## Technical Questions

### What technologies does this application use?
- **Frontend:** React, Material UI, Monaco Editor (for code display/editing).
- **Backend:** Python, Flask, Google Agent Development Kit (ADK), potentially Google Cloud Client Libraries.
- **Communication:** Frontend calls backend API endpoints via HTTP (fetch).

### Does the application work offline?
- **Frontend GUI:** The interface can be loaded offline if cached, but creating/editing configurations is the main purpose.
- **Backend Service:** Requires an internet connection to interact with Google Cloud APIs (LLMs, Search, etc.).
- **Testing Agents:** Requires both frontend and backend to be running and the backend to have internet access.

### Where is my data stored?
- **Agent Configurations:** All configurations you create in the GUI are stored locally in your **browser's localStorage**.
- **API Key (GUI):** The Google Cloud API key entered for validation is stored in **browser localStorage**.
- **Backend Credentials:** Handled by the backend service via environment variables (`.env` file or system environment) - **not stored by the GUI**.
- **Agent Execution Data:** Conversation history and state during a run are managed by the ADK's `SessionService` (using `InMemorySessionService` by default in the provided backend code, meaning history is lost when the backend restarts unless a persistent service is configured).

### Can I self-host the Gemini Agent Builder GUI and Backend?
Yes. Clone the repository, set up the frontend (`npm install`) and backend (`pip install`, configure `.env`), and run both services concurrently. See the [deployment guide](./deployment.md) for production hosting.

## Troubleshooting

### API Key Validation Error in GUI
- Ensure the key is a valid Google Cloud API Key.
- Check the **backend terminal logs** for specific error messages from the `/api/validate-key` endpoint.
- Verify the backend's credentials (in `.env`) are correct and have basic permissions (like listing resources) in your GCP project.
- Ensure required Google Cloud APIs (e.g., Vertex AI) are enabled in your project.

### Agent Run Errors / No Response in Test Chat
- Check the **backend terminal logs** first. This is where ADK execution errors and detailed logs will appear.
- Ensure the Flask backend (`app.py`) is running.
- Verify backend credentials in `.env` are correct and have permissions for the models and tools the agent uses (e.g., Vertex AI User role).
- Check agent instructions: Are they clear? Do they correctly reference tool names (e.g., `google_search`, `call_db_agent`)?
- Built-in Tool Limitations: Remember only one built-in tool per agent is currently supported by ADK.
- Check the Google Cloud status page for any service disruptions.

### Backend Connection Issues (Frontend Error)
- Ensure the Flask backend is running on the expected port (default 5001).
- Check browser console for CORS errors. If deploying frontend/backend separately, ensure backend CORS is configured correctly.
- Verify the `BACKEND_URL` in `src/utils/apiService.js` is correct.

### Hitting Google Cloud API Rate Limits
- Monitor your API usage in the Google Cloud Console.
- Implement caching in the backend for repetitive tool calls if applicable.
- Consider requesting quota increases from Google Cloud if necessary.

## Contributing and Support

### How can I contribute?
- Report bugs or suggest features via GitHub Issues.
- Submit Pull Requests with code improvements or new features (update both frontend and backend if needed).
- Improve documentation.
- See [CONTRIBUTING.md](../CONTRIBUTING.md).

### Where can I get help?
- Check this FAQ and other documentation files.
- Review the [Google ADK Reference Notes](./Google-ADK%20Reference%20Notes.md).
- Consult the [Official Google ADK Documentation](https://google.github.io/adk-docs/).
- Open a GitHub Issue for specific problems with *this GUI application*. For issues with the ADK itself, refer to the official ADK GitHub repository.

### Are there tutorials or examples?
Yes, check the [Example Agent Configurations](./example-agents.md) page for guidance on configuring different types of agents using the GUI. The generated code snippets also serve as examples.
```
*Self-correction:* Updated all relevant answers to refer to Google ADK, Gemini, Google Cloud credentials (API Key vs Service Account), the Flask backend architecture, local storage usage, Google Cloud pricing/status, and ADK-specific troubleshooting points (backend logs, credential permissions, built-in tool limits).

---

This chunk updates the core documentation files (`Google-ADK Reference Notes.md`, test scripts, `docs/index.md`, `docs/getting-started.md`, `docs/deployment.md`, `docs/example-agents.md`, `docs/faq.md`) to align with the Google ADK, the Flask backend, and the information provided in `google-sdk-docs.md`.

We are nearing the end of the refactoring. The main parts left are potentially updating the `LICENSE`, `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`, and the `.github` issue/PR templates to reflect the project's new name and context, although these are less critical for functionality.

Please let me know if this chunk is approved.