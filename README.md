
# Gemini Agent Builder GUI (using Google ADK)

![Google ADK](https://img.shields.io/badge/Google-Agent%20Development%20Kit%20(ADK)-4285F4?style=for-the-badge&logo=google&logoColor=white)
![React](https://img.shields.io/badge/React-18.2.0-61DAFB?style=for-the-badge&logo=react&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![Material UI](https://img.shields.io/badge/Material%20UI-5.15.0-0081CB?style=for-the-badge&logo=mui&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

A professional, intuitive GUI application paired with a Flask backend for creating configurations for, testing, and managing AI agents using **Google's Agent Development Kit (ADK)**. Build powerful AI agents powered by Gemini models with custom tools, sub-agent interactions (agent-as-tool), and callbacks without needing deep ADK expertise for configuration.

> **IMPORTANT**: This project is created by Gregory Kennedy (https://aiforsocialbenefit.xyz | peacedude@gmail.com) and is **not affiliated with, endorsed by, or sponsored by Google**. It is an independent tool designed to facilitate working with the **Google Agent Development Kit (ADK)**.

## 🌟 Features

### Agent Configuration & Management
- **Intuitive Agent Config Builder**: Step-by-step wizard interface for defining agent configurations.
- **Visual Tool Configuration**: Add and configure built-in Google ADK tools (like Google Search, Code Execution) and define custom Python function tools.
- **Agent-as-Tool Configuration**: Define how agents can call other agents.
- **Agent Config Dashboard**: Manage all your saved agent configurations in one place.
- **Clone & Edit**: Easily duplicate and modify existing agent configurations.

### Advanced Capabilities Configuration
- **Custom Instructions**: Create detailed system prompts using Gemini models.
- **Model Selection**: Choose from Google's Gemini models (e.g., `gemini-2.5-flash-preview-04-17`, `gemini-2.5-pro-preview-05-06`).
- **Tool Integration**: Configure Google Search, Code Execution, Vertex AI Search, and custom Python function tools.
- **Callback Configuration**: Define Python functions to hook into agent lifecycle events (e.g., before/after model calls, before/after tool calls) for observation, control, or implementing guardrails.
- **Structured Output**: Define Pydantic schemas for agent responses.

### Testing & Deployment
- **Built-in Test Environment**: Interact with your agent configurations via the Flask backend which runs the actual ADK agent.
- **Real-time Chat Interface**: See agent responses, tool calls (simulated based on backend execution), and callback triggers (logged in backend).
- **Code Generation**: Automatically generate Python code snippets for defining agents and tools using Google ADK based on your configuration.
- **Export Options**: Download or copy generated Python code for use in your ADK applications.

### User Experience
- **Dark/Light Mode**: Toggle between dark and light themes.
- **Responsive Design**: Works on desktop and tablet devices.
- **Local Storage**: Agent configurations are saved locally in your browser for privacy and convenience.
- **API Key Management**: Securely store your Google Cloud API key locally; validation and usage occur via the backend service.

## 📸 Screenshots

Here are some glimpses of the Gemini Agent Builder GUI:

![Screenshot 1: Agent Builder Dashboard](public/1-image-gemini-agent-builder-1200x1000p.png)

*Agent Configuration Dashboard*

![Screenshot 2: Agent Builder Step 1](public/2-image-gemini-agent-builder-1200x1000p.png)

*Agent Builder - Basic Details Step*

![Screenshot 3: Agent Builder Step 2](public/3-image-gemini-agent-builder-1200x1000p.png)

*Agent Builder - Instructions Editor*

![Screenshot 4: Agent Builder Step 3](public/4-image-gemini-agent-builder-1200x1000p.png)

*Agent Builder - Tools Selector*

![Screenshot 5: Agent Test Chat](public/app-react-terminal-gemini-agent-builder.png)

*Agent Testing Interface*
## � Getting Started

### Prerequisites

- Node.js (v16.x or higher recommended)
- npm (v8.x or higher recommended)
- Python (v3.12 or higher recommended for ADK)
- pip (Python package installer)
- Google Cloud API Key with necessary permissions (e.g., Vertex AI API enabled). You will also need a Google Cloud Project if you intend to use models or services hosted on Vertex AI.
- (Optional but Recommended) Google Cloud Service Account Key for the backend, especially for more secure or production-like local testing.

### Installation

**1. Clone the Repository**
```bash
git clone https://github.com/yourusername/gemini-agentsdk-gui-app.git # Use the new repo name
cd gemini-agentsdk-gui-app```

**2. Set up Frontend**
```bash
# Navigate to the root directory if not already there
pnpm install
```

**3. Set up Backend**
```bash
cd backend
# Create a Conda virtual environment (recommended)
# Make sure you have Anaconda or Miniconda installed.
# Replace 'gemini-adk-env' with your preferred environment name.
# Replace '3.12' with your desired Python version if different (ADK recommends 3.12+).
conda create --name gemini-adk-env python=3.12 -y
conda activate gemini-adk-env

# Install backend dependencies
pip install -r requirements.txt

# Configure Google Cloud Credentials
# These credentials are used by the local Flask backend to communicate with Google Cloud services,
# including accessing Gemini models directly or via Vertex AI.
#
# 1. Rename .env.example to .env
# 2. Edit .env:
#    - Add your GOOGLE_API_KEY (for direct Gemini API access).
#    - OR set GOOGLE_APPLICATION_CREDENTIALS to the path of your service account key JSON file (recommended for Vertex AI and other GCP services).
#    - Optionally set GOOGLE_PROJECT_ID if using Vertex AI or other project-specific GCP services.
#
# Example using API Key in .env:
# GOOGLE_API_KEY=AIzaSy..........
#
# Example using Service Account Key in .env:
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
# GOOGLE_PROJECT_ID=your-gcp-project-id

# To deactivate the Conda environment when you're done:
# conda deactivate

cd .. # Go back to the root directory
```

**4. Run the Application**

You need to run both the frontend and the backend concurrently.

*   **Terminal 1: Start Backend (Flask)**
    ```bash
    cd backend
    conda activate gemini-adk-env # Activate Conda env if not already active
    flask run --port 5001 # Or python app.py if flask run doesn't work
    ```
    *(Keep this terminal running)*

*   **Terminal 2: Start Frontend (React)**
    ```bash
    # In the root directory
    npm start
    ```

**5. Access the GUI**
Open your browser and navigate to `http://localhost:3000` (or the port specified by React).

### Building for Production

*   **Frontend:**
    ```bash
    npm run build
    ```
    The build files will be in the `build/` directory. Serve these static files using a web server (like Nginx, Caddy, or a cloud service).

*   **Backend:**
    The Flask backend can be run locally (as described in "Run the Application") for development and testing.
    For a production setup or to run the backend within GCP (e.g., for closer integration with Vertex AI services):
    - Deploy the Flask application to a suitable GCP service like Cloud Run, App Engine, or Google Kubernetes Engine (GKE).
    - When deployed on GCP, it's best practice to use GCP's native authentication mechanisms (e.g., service account identity associated with the Cloud Run service) instead of embedding API keys or credential files. The Google ADK and client libraries will automatically pick up these credentials.
    - Use a production-ready WSGI server (like Gunicorn or Waitress) behind a reverse proxy (like Nginx or Google Cloud Load Balancing).
    - Ensure necessary environment variables (like `GOOGLE_PROJECT_ID` if not inferred) are configured for the deployed service.
    - See `docs/deployment.md` for more general deployment considerations.

## 📖 Usage Guide

### Setting Up Your API Key

1.  When you first launch the application, you'll be prompted to enter your Google Cloud API key.
2.  Enter the key and click "Continue". The key is sent to the backend for validation.
3.  If valid, the key is stored *only* in your browser's local storage for the frontend to know validation passed. The backend uses credentials from its environment variables (`.env` or system environment) for actual API calls.
4.  You can test your Google Cloud API connectivity using the included test script (adapt as needed):
    ```bash
    # Example - script needs refinement for Google Cloud
    ./test-google-api.bat
    ```

### Creating Your First Agent Configuration

1.  Click the "Create Agent Config" button on the dashboard.
2.  Follow the step-by-step wizard:
    *   **Basic Details**: Set config name, description, and select a Gemini model. Configure model settings (temperature, TopP, TopK, etc.).
    *   **Instructions**: Define the agent's system prompt, guiding its behavior, persona, and task execution.
    *   **Tools**: Add built-in Google ADK tools (Google Search, Code Execution) or define custom Python function tools.
    *   **Sub-Agents (Agent-as-Tool)**: Configure references to other agent configurations that this agent can call as tools.
    *   **Guardrails (Callbacks)**: Configure Python callback functions to observe or modify agent behavior at specific lifecycle points (placeholder UI, requires backend implementation).
    *   **Code Preview**: View and copy/download the generated Python code (using Google ADK) based on your configuration.
    *   **Test Agent Config**: Interact with your agent configuration via the chat interface, which communicates with the backend Flask service running the ADK agent.

### Testing Your Agent Configuration

1.  Use the built-in test environment ("Test Agent Config" tab or "Test" button on the dashboard).
2.  Type messages and observe the agent's responses generated by the backend ADK execution.
3.  Check the backend terminal logs for detailed execution flow, tool calls, and callback triggers.
4.  Refine your agent configuration (instructions, tools, etc.) based on test results and save the changes.

### Deploying Your Agent

1.  Generate the Python code for your agent configuration using the Code Preview tab.
2.  Copy or download the `agent.py`, `instructions.py`, `tools.py` (if applicable) files.
3.  Use this code in your own Python application with the Google ADK installed.
4.  Set up the necessary Google Cloud credentials in your deployment environment.
5.  Use the Google ADK `Runner` (as shown in the generated runner/streaming examples) to interact with your agent.
6.  Refer to the official Google Agent Development Kit (ADK) documentation for advanced usage and deployment patterns.

## 🏗️ Architecture

This application now uses a client-server architecture:

### Core Components (Google ADK - Run by Backend)

- **Agent (google.adk.Agent)**: Core component defined in Python, powered by an LLM (Gemini). Uses instructions, tools, and potentially sub-agents/callbacks.
- **Runner (google.adk.runners.Runner)**: Executes agent invocations, manages the event loop, and interacts with services.
- **Tools (google.adk.tools...)**: Built-in capabilities (Google Search, Code Execution) or custom Python functions (FunctionTool, LongRunningFunctionTool) or other agents (AgentTool).
- **Callbacks (Python functions)**: Hooks into the agent lifecycle for observation or control.
- **SessionService (google.adk.sessions...)**: Manages conversation state and history.
- **InvocationContext / CallbackContext / ToolContext**: Provides context to execution logic and callbacks.

### Application Structure

- **React Frontend**: Built with React and Material UI for the user interface to *configure* agents.
- **Local Storage**: Agent *configurations* and UI settings are stored in the browser.
- **Flask Backend**: Python server using Flask.
    - Receives requests from the frontend (validate API key, run agent).
    - Instantiates and runs agents using the **Google ADK** based on the received configuration.
    - Interacts with Google AI services (like Gemini models via direct API or Vertex AI) using the configured backend credentials (local `.env` or GCP environment credentials if deployed on GCP).
    - Returns results to the frontend.
- **API Communication**: Frontend communicates with the Flask backend via HTTP requests (fetch/axios).

## 🔧 Implementation Details

Key files involved in the refactoring:

- **`backend/app.py`**: The Flask application handling API requests and running ADK agents.
- **`backend/requirements.txt`**: Python dependencies for the backend (Flask, Google ADK, etc.).
- **`backend/.env`**: Stores Google Cloud credentials for the backend.
- **`src/utils/apiService.js`**: Frontend service making API calls to the Flask backend.
- **`src/components/AgentBuilder.js`**: Main UI component for configuring agents.
- **`src/components/builder/*`**: Sub-components for specific configuration steps (Tools, Instructions, SubAgents, Callbacks).
- **`src/components/AgentTester.js` / `src/components/builder/TestAgent.js`**: UI for interacting with the agent via the backend.
- **`src/utils/codeTemplates.js`**: Python code templates updated for Google ADK syntax.
- **`src/utils/tools.js`**: Definitions and examples of Google ADK built-in and custom tools.
- **`src/utils/modelOptions.js`**: Updated list of Google AI models (Gemini).
- **Documentation (`README.md`, `docs/*`)**: Updated to reflect Google ADK and the new architecture.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/amazing-feature`).
3.  Make your changes in both frontend and backend code as needed.
4.  Commit your changes (`git commit -m 'Add some amazing feature'`).
5.  Push to the branch (`git push origin feature/amazing-feature`).
6.  Open a Pull Request against the main branch.
7.  Ensure your changes follow the project's coding style and include relevant tests if applicable.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. Copyright (c) 2025 J. Gravelle.

## 🙏 Acknowledgments

- Built with React and Material UI.
- Powered by **Google's Agent Development Kit (ADK)** and **Gemini Models**.
- Backend implemented with Flask.
- Inspired by the need for accessible tools to configure and test sophisticated AI agents.

## 🔗 Links

- **Google Agent Development Kit (ADK) GitHub:** [https://github.com/google/adk-python](https://github.com/google/adk-python) (Link to main ADK repo)
- **Google ADK Documentation:** [https://google.github.io/adk-docs/](https://google.github.io/adk-docs/) (Link to ADK docs site)
- **Google AI for Developers:** [https://ai.google.dev/](https://ai.google.dev/)
- **Vertex AI Documentation:** [https://cloud.google.com/vertex-ai/docs](https://cloud.google.com/vertex-ai/docs)
- [React Documentation](https://reactjs.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Material UI Documentation](https://mui.com/)

---

<p align="center">
  <i>Configure and test powerful Google ADK agents without writing all the boilerplate</i><br>
  <a href="https://github.com/yourusername/gemini-agentsdk-gui-app/issues">Report Bug</a> ·
  <a href="https://github.com/yourusername/gemini-agentsdk-gui-app/issues">Request Feature</a>
</p>



