
---

**`docs/deployment.md` (Modified)**

```markdown
# Deployment Guide - Gemini Agent Builder GUI

This guide provides instructions for deploying the **Gemini Agent Builder GUI** (React frontend) and its associated **Flask backend service**, as well as the agent configurations you create.

**IMPORTANT**: This project is created by Gregory Kennedy (https://aiforsocialbenefit.xyz | peacedude@gmail.com) and is **not affiliated with, endorsed by, or sponsored by Google**. It is an independent tool designed to facilitate working with the **Google Agent Development Kit (ADK)**.

## Deploying the Application Components

The application consists of two main parts that need to be deployed:

1.  **React Frontend (GUI):** A static web application built with React.
2.  **Flask Backend (ADK Runner):** A Python service that runs the Google ADK agents.

### Deploying the React Frontend (GUI)

The frontend is a standard React application and can be deployed using various methods for static sites.

**1. Build the Frontend:**
From the project's root directory, run the build command:
```bash
npm run build
```
*(If you use `pnpm`, use `pnpm run build`)*

**2. Choose a Deployment Option:**

*   **Option 1: Static Site Hosting Services**
    *   The build files will be in the `build/` directory.
    *   Deploy these files to services like:
        *   Firebase Hosting (integrates well with Google Cloud)
        *   Google Cloud Storage (with Load Balancer or Cloudflare)
        *   Netlify
        *   Vercel
        *   GitHub Pages
        *   AWS S3 + CloudFront
        *   Azure Static Web Apps

*   **Option 2: Docker Container (Frontend Only)**
    *   Create a `Dockerfile` specifically for the frontend (e.g., using a multi-stage build with Node.js to build and Nginx/Caddy to serve).
    ```dockerfile
    # Example Dockerfile for React Frontend (using Nginx)

    # Stage 1: Build the React app
    FROM node:18-alpine as builder
    WORKDIR /app
    COPY package.json pnpm-lock.yaml ./
    # Use pnpm if specified, otherwise npm
    # RUN pnpm install
    RUN npm install
    COPY . .
    RUN npm run build

    # Stage 2: Serve using Nginx
    FROM nginx:stable-alpine
    COPY --from=builder /app/build /usr/share/nginx/html
    # Copy a custom nginx config if needed (e.g., to handle routing)
    # COPY nginx.conf /etc/nginx/conf.d/default.conf
    EXPOSE 80
    CMD ["nginx", "-g", "daemon off;"]
    ```
    *   Build the image: `docker build -t gemini-agent-builder-frontend .`
    *   Run the container: `docker run -p 8080:80 gemini-agent-builder-frontend` (Access at `http://localhost:8080`)

*   **Option 3: Self-Hosted Web Server**
    *   Serve the contents of the `build/` directory using a web server like Nginx or Apache.
    *   Ensure the server is configured to handle single-page application routing (redirect all non-file requests to `index.html`).
    *   Example Nginx configuration snippet:
        ```nginx
        server {
            listen 80;
            server_name your-frontend-domain.com;
            root /path/to/your/project/build; # Path to the build directory
            index index.html;

            location / {
                try_files $uri $uri/ /index.html; # Handle SPA routing
            }

            # Add configuration to proxy API requests to the backend if needed
            # location /api/ {
            #    proxy_pass http://your_backend_service_address:5001;
            #    proxy_set_header Host $host;
            #    proxy_set_header X-Real-IP $remote_addr;
            #    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            # }
        }
        ```

### Deploying the Flask Backend (ADK Runner)

The backend requires a Python environment with the necessary dependencies and access to Google Cloud credentials.

**1. Prepare the Backend Code:**
Ensure the `backend/` directory contains `app.py`, `requirements.txt`, and your `.env` file (or that environment variables are set in the deployment environment).

**2. Choose a Deployment Option:**

*   **Option 1: Platform-as-a-Service (PaaS)**
    *   Services like **Google Cloud Run**, **Google App Engine**, or Heroku are suitable.
    *   **Google Cloud Run (Recommended):**
        *   Containerize the backend (see Docker option below).
        *   Push the container image to Google Artifact Registry.
        *   Deploy the image to Cloud Run.
        *   Configure environment variables directly in Cloud Run (for `GOOGLE_APPLICATION_CREDENTIALS` path within the container or `GOOGLE_API_KEY`, `GOOGLE_PROJECT_ID`).
        *   Ensure the Cloud Run service account has the necessary IAM permissions (e.g., Vertex AI User).

*   **Option 2: Docker Container (Backend)**
    *   Create a `Dockerfile` in the `backend/` directory:
    ```dockerfile
    # backend/Dockerfile
    FROM python:3.10-slim # Choose appropriate Python version

    WORKDIR /app

    # Copy only requirements first to leverage Docker cache
    COPY requirements.txt requirements.txt
    RUN pip install --no-cache-dir --upgrade pip && \
        pip install --no-cache-dir -r requirements.txt

    # Copy the rest of the backend code
    COPY . .

    # Set environment variable for Flask app (optional, can be set at runtime)
    ENV FLASK_APP=app.py
    ENV FLASK_RUN_HOST=0.0.0.0
    ENV FLASK_RUN_PORT=5001

    # Expose the port Flask runs on
    EXPOSE 5001

    # Command to run the application using a production server like Gunicorn
    # Adjust workers as needed
    CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]
    ```
    *   Build the image: `docker build -t gemini-agent-builder-backend ./backend`
    *   Run the container, mounting credentials or setting environment variables:
        ```bash
        # Example using service account file mounted as volume
        docker run -p 5001:5001 \
          -v /path/to/your/keyfile.json:/app/keyfile.json \
          -e GOOGLE_APPLICATION_CREDENTIALS=/app/keyfile.json \
          -e GOOGLE_PROJECT_ID=your-gcp-project-id \
          gemini-agent-builder-backend

        # Example using API Key environment variable
        # docker run -p 5001:5001 \
        #  -e GOOGLE_API_KEY=AIzaSy... \
        #  -e GOOGLE_PROJECT_ID=your-gcp-project-id \
        #  gemini-agent-builder-backend
        ```

*   **Option 3: Virtual Machine / Server**
    *   Provision a server (e.g., Google Compute Engine, AWS EC2).
    *   Install Python and dependencies (`pip install -r requirements.txt`).
    *   Configure Google Cloud credentials (e.g., using `gcloud auth application-default login` or setting `GOOGLE_APPLICATION_CREDENTIALS`).
    *   Run the Flask app using a production WSGI server (Gunicorn, Waitress) often managed by a process manager (like `systemd` or `supervisor`).
    *   Use a reverse proxy (Nginx, Apache) in front of the WSGI server for handling HTTPS, static files (if any), and load balancing.

**Important Backend Configuration:**
*   **Credentials:** Securely provide Google Cloud credentials to the backend environment. Using **Service Accounts** (`GOOGLE_APPLICATION_CREDENTIALS`) is strongly recommended for security and manageability over API keys in backend services.
*   **CORS:** Ensure the Flask backend is configured with CORS to allow requests from your deployed frontend's domain. The provided `app.py` includes basic CORS setup (`flask_cors`). You might need to restrict origins in production.
*   **Environment Variables:** Set `GOOGLE_PROJECT_ID` if needed by client libraries.

## Deploying Agent Configurations Created with the Builder

The GUI helps you *configure* agents and generate the corresponding Python code using Google ADK. You then deploy this generated code.

### Option 1: Standalone Python Script/Application

1.  Use the "Code Preview" tab in the GUI to get the generated Python code (e.g., `agent.py`).
2.  Create your Python project structure.
3.  Install the Google ADK and other dependencies:
    ```bash
    pip install google-adk yfinance # Add other tool dependencies
    ```
4.  Set up Google Cloud Authentication in your environment (the recommended way is setting the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of your service account key file).
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
    # Optionally, set project ID if needed by libraries
    # export GOOGLE_PROJECT_ID="your-gcp-project-id"
    ```
5.  Use the Google ADK `Runner` to interact with your agent definition as shown in the generated runner/streaming examples.
    ```python
    # Example: run_my_agent.py
    import asyncio
    from agent import agent # Assuming agent definition is in agent.py
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    import os

    # Ensure credentials are configured via environment variables

    APP_NAME = "my_deployed_agent_app"
    USER_ID = "deployed_user_1"
    SESSION_ID = "deployed_session_1"

    async def main():
        session_service = InMemorySessionService()
        session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
        runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

        query = "What is the stock price for GOOG?" # Example query
        content = types.Content(role='user', parts=[types.Part(text=query)])

        print(f"Running agent with query: {query}")
        async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
            if event.is_final_response() and event.content and event.content.parts:
                print("Agent Response:", event.content.parts.text)
            # Add more event handling if needed (tool calls, etc.)

    if __name__ == "__main__":
        asyncio.run(main())
    ```
6.  Run your script: `python run_my_agent.py`

### Option 2: Deploy as a Web API (e.g., using Flask/FastAPI)

Expose your agent configuration via its own API endpoint.

1.  Create a new Flask or FastAPI application.
2.  Include your generated agent definition code (`agent.py`, `tools.py`, etc.).
3.  Install dependencies (`google-adk`, `flask`, etc.).
4.  Set up Google Cloud credentials in the API's environment.
5.  Create an endpoint that takes user input, initializes the ADK `Runner` and `SessionService`, runs the agent using `runner.run_async`, and returns the final response.

    ```python
    # Example: agent_api.py (Flask)
    from flask import Flask, request, jsonify
    import asyncio
    from agent import agent # Your generated agent definition
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService # Or a persistent one
    from google.genai import types
    import os
    import uuid

    # Ensure credentials are set in environment

    app = Flask(__name__)
    session_service = InMemorySessionService() # Use a persistent service in production
    APP_NAME = "my_agent_api"

    @app.route('/api/chat', methods=['POST'])
    async def chat():
        data = request.json
        user_input = data.get('message', '')
        user_id = data.get('user_id', 'api_user') # Get user/session from request
        session_id = data.get('session_id', str(uuid.uuid4())) # Generate or get session ID

        if not user_input:
            return jsonify({'error': 'No message provided'}), 400

        try:
            # Ensure session exists
            try:
                session = session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
            except KeyError:
                session = session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)

            runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
            content = types.Content(role='user', parts=[types.Part(text=user_input)])
            final_response_text = "Agent did not produce a final text response."

            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
                if event.is_final_response() and event.content and event.content.parts:
                    final_response_text = event.content.parts.text

            return jsonify({
                'response': final_response_text,
                'user_id': user_id,
                'session_id': session_id
            })
        except Exception as e:
            print(f"Error running agent via API: {e}")
            return jsonify({'error': f'Agent execution failed: {str(e)}'}), 500

    if __name__ == '__main__':
        # Use a production WSGI server like gunicorn instead of app.run in production
        app.run(debug=False, port=5002)
    ```
6.  Deploy this API application using methods similar to the backend deployment (Cloud Run, VM, etc.).

### Option 3: Integration into Existing Applications

Integrate the generated agent code directly into your existing Python application.

1.  Install `google-adk` and other dependencies in your application's environment.
2.  Copy the generated agent definition code into your project.
3.  Ensure Google Cloud credentials are available to your application.
4.  Import the agent definition.
5.  Instantiate the ADK `Runner` and `SessionService` where needed in your application logic.
6.  Call `runner.run_async` with user input and handle the resulting events or final response.

## Production Considerations

### Credential Management (Critical)
*   **Backend Service:** Use **Google Cloud Service Accounts** (`GOOGLE_APPLICATION_CREDENTIALS`) for the Flask backend. Grant least privilege IAM roles. Store keys securely (e.g., Secret Manager) and inject them as environment variables.
*   **Standalone Scripts/APIs:** Use Service Accounts or Application Default Credentials (`gcloud auth application-default login` for development, service accounts for production).
*   **Avoid API Keys** for backend services or deployed applications whenever possible due to security risks. Never hardcode credentials.

### Rate Limiting and Quotas
*   Be aware of Google Cloud API (e.g., Vertex AI Gemini API) rate limits and quotas associated with your project and credentials.
*   Implement appropriate error handling and retry logic (e.g., exponential backoff) for rate limit errors (HTTP 429).
*   Monitor usage via the Google Cloud Console.

### Error Handling
*   Implement robust error handling in the Flask backend and any deployed agent applications.
*   Catch exceptions during ADK execution (`runner.run_async`), Google API calls, and tool execution.
*   Log errors effectively (e.g., using Google Cloud Logging).
*   Provide informative error responses to the frontend or calling application.

### Monitoring and Logging
*   Utilize **Google Cloud Logging** and **Monitoring** for the backend service and deployed agents.
*   Log key events: agent invocations, tool calls/results, errors, final responses.
*   Monitor resource usage (CPU, memory), latency, and error rates.
*   Set up alerts for critical errors or quota limits.

### Security Considerations
*   Secure the Flask backend endpoint (HTTPS, authentication/authorization if needed beyond internal use).
*   Validate and sanitize any input passed to agents or tools, especially if using code execution or tools interacting with external systems.
*   Regularly update dependencies (`pip install --upgrade ...`) for both the backend and deployed agent applications.
*   Review IAM permissions for service accounts regularly.

### Scaling Considerations
*   **Backend Service:** Deploy the Flask backend using scalable solutions like Google Cloud Run (auto-scaling), Google Kubernetes Engine (GKE), or App Engine. Use load balancing if needed.
*   **Session Management:** For high scalability, replace `InMemorySessionService` with a persistent, scalable solution (e.g., using Firestore, Memorystore, or a database via a custom `BaseSessionService` implementation).
*   **Agent Design:** Optimize agent instructions and tool usage for efficiency.

### Cost Optimization
*   Choose the most cost-effective **Gemini model** suitable for the agent's task complexity.
*   Monitor Google Cloud billing and set budgets/alerts.
*   Optimize prompts and instructions to minimize token usage per invocation.
*   Implement caching for tool results or agent responses where appropriate (requires careful state management).
```

---
