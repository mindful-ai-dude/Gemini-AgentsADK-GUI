
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
    # Copy package files and install dependencies (use pnpm if needed)
    COPY package.json package-lock.json* pnpm-lock.yaml* ./
    # RUN pnpm install --frozen-lockfile
    RUN npm ci
    COPY . .
    RUN npm run build

    # Stage 2: Serve using Nginx
    FROM nginx:stable-alpine
    COPY --from=builder /app/build /usr/share/nginx/html
    # Optional: Copy a custom nginx config for SPA routing
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

            # Proxy API requests to the backend service
            location /api/ {
               proxy_pass http://<your_backend_service_address>:5001; # Point to backend
               proxy_set_header Host $host;
               proxy_set_header X-Real-IP $remote_addr;
               proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
               proxy_set_header X-Forwarded-Proto $scheme;
            }
            # Add SSL configuration for HTTPS
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
    *   Create a `Dockerfile` in the `backend/` directory (refer to the example provided in Chunk 5, or adapt the one below):
    ```dockerfile
    # backend/Dockerfile
    FROM python:3.11-slim # Use a recent, stable Python version compatible with ADK

    WORKDIR /app

    # Install OS packages if needed (e.g., for specific libraries)
    # RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

    # Copy only requirements first to leverage Docker cache
    COPY requirements.txt requirements.txt
    RUN pip install --no-cache-dir --upgrade pip && \
        pip install --no-cache-dir -r requirements.txt

    # Copy the rest of the backend code
    COPY . .

    # Set environment variable for Flask app (can be overridden at runtime)
    ENV FLASK_APP=app.py
    ENV FLASK_RUN_HOST=0.0.0.0
    ENV FLASK_RUN_PORT=8080 # Use standard 8080 for Cloud Run compatibility

    EXPOSE 8080

    # Use Gunicorn as the production WSGI server
    CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "app:app"]
    ```
    *   Build the image: `docker build -t gemini-agent-builder-backend ./backend`
    *   Run the container, mounting credentials or setting environment variables:
        ```bash
        # Example using service account file mounted as volume
        docker run -p 5001:8080 \
          -v /path/to/your/keyfile.json:/app/keyfile.json:ro \
          -e GOOGLE_APPLICATION_CREDENTIALS=/app/keyfile.json \
          -e GOOGLE_PROJECT_ID=your-gcp-project-id \
          gemini-agent-builder-backend

        # Example using API Key environment variable (less recommended for backend)
        # docker run -p 5001:8080 \
        #  -e GOOGLE_API_KEY=AIzaSy... \
        #  -e GOOGLE_PROJECT_ID=your-gcp-project-id \
        #  gemini-agent-builder-backend
        ```

*   **Option 3: Virtual Machine / Server**
    *   Provision a server (e.g., Google Compute Engine, AWS EC2).
    *   Install Python and dependencies (`pip install -r backend/requirements.txt`).
    *   Configure Google Cloud credentials (e.g., using `gcloud auth application-default login` or setting `GOOGLE_APPLICATION_CREDENTIALS`).
    *   Run the Flask app using a production WSGI server (Gunicorn, Waitress) managed by a process manager (like `systemd` or `supervisor`).
    *   Use a reverse proxy (Nginx, Apache) in front of the WSGI server.

**Important Backend Configuration:**
*   **Credentials:** Use **Service Accounts** (`GOOGLE_APPLICATION_CREDENTIALS`) for security and manageability. Grant least privilege IAM roles.
*   **CORS:** Ensure the Flask backend's CORS configuration in `backend/app.py` allows requests *only* from your deployed frontend's domain in production (`CORS(app, origins=["https://your-frontend-domain.com"])`).
*   **Environment Variables:** Set `GOOGLE_PROJECT_ID`, `GOOGLE_APPLICATION_CREDENTIALS` (or `GOOGLE_API_KEY`), and potentially other necessary variables (e.g., database URLs if using persistent session storage).

## Deploying Agent Configurations Created with the Builder

The GUI generates Python code using Google ADK. You deploy this *code*, not the GUI configuration itself.

### Option 1: Standalone Python Script/Application

1.  Generate the Python code (`agent.py`, etc.) from the "Code Preview" tab.
2.  Set up a Python project, install dependencies (`pip install google-adk ...`).
3.  Configure Google Cloud Authentication (preferably `GOOGLE_APPLICATION_CREDENTIALS`).
4.  Use the ADK `Runner` as shown in the generated examples to interact with your `agent` definition.
5.  Run your script (`python your_script.py`).

### Option 2: Deploy as a Web API (e.g., using Flask/FastAPI)

Expose your generated agent code via its own dedicated API.

1.  Create a new Flask/FastAPI project.
2.  Include your generated agent definition code.
3.  Install dependencies.
4.  Set up credentials.
5.  Create API endpoints that instantiate the ADK `Runner` and `SessionService` (consider persistent session service like `DatabaseSessionService`), run the agent using `runner.run_async`, and return the result.
6.  Deploy this API application (Cloud Run, VM, etc.).

### Option 3: Integration into Existing Applications

Integrate the generated agent code into your existing Python application.

1.  Install `google-adk` in your application environment.
2.  Copy the generated agent code.
3.  Configure credentials.
4.  Import the agent definition and use the ADK `Runner` within your application logic.

## Production Considerations

### Credential Management (Critical)
*   **Backend Service:** Use **Google Cloud Service Accounts** (`GOOGLE_APPLICATION_CREDENTIALS`). Store keys securely (e.g., Secret Manager). Grant least privilege.
*   **Standalone Apps/APIs:** Use Service Accounts or ADC.
*   **Avoid API Keys** for backend services. Never hardcode credentials.

### Rate Limiting and Quotas
*   Monitor Google Cloud API quotas (Vertex AI, etc.).
*   Implement retry logic (exponential backoff) for `429` errors in your backend or agent runner logic.

### Error Handling & Logging
*   Implement robust error handling in the Flask backend and deployed agent code (catch ADK/API exceptions).
*   Use Google Cloud Logging for structured logs from the backend and deployed agents. Log invocations, errors, tool usage.

### Monitoring
*   Use Google Cloud Monitoring for backend performance (latency, errors, resource usage). Set alerts.

### Security
*   Secure the Flask backend endpoint (HTTPS, firewall, potentially authentication if not internal).
*   Sanitize inputs passed to agents, especially if using code execution or external tools.
*   Regularly update dependencies (`pip install --upgrade ...`).
*   Review IAM permissions.

### Scaling
*   **Backend Service:** Use scalable platforms (Cloud Run, GKE, App Engine).
*   **Session Management:** Use a persistent `SessionService` (`DatabaseSessionService` or `VertexAiSessionService`) instead of `InMemorySessionService` for production to handle multiple instances and state persistence.

### Cost Optimization
*   Choose appropriate Gemini models.
*   Monitor Google Cloud billing.
*   Optimize prompts/instructions for token efficiency.
*   Consider caching where applicable.
```



