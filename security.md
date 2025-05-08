# Security Policy - Gemini Agent Builder GUI

**IMPORTANT**: This project is created by Gregory Kennedy (https://aiforsocialbenefit.xyz | peacedude@gmail.com) and is **not affiliated with, endorsed by, or sponsored by Google**. It is an independent tool designed to facilitate working with the **Google Agent Development Kit (ADK)**.

## Reporting a Vulnerability

We take the security of this application seriously. If you believe you've found a security vulnerability in the **frontend GUI** or the **backend Flask service**, please follow these steps:

1.  **Do not disclose the vulnerability publicly** until it has been addressed.
2.  Email details to  Gregory Kennedy (peacedude@gmail.com).
3.  Include:
    *   A description of the vulnerability.
    *   Steps to reproduce (mention if it affects frontend, backend, or both).
    *   Potential impact.
    *   Suggested fixes (if any).

## Response Process

1.  Acknowledgement within 48 hours.
2.  Investigation of impact and severity.
3.  Development and testing of a fix.
4.  Publication of a new release addressing the vulnerability.
5.  Public disclosure after a reasonable update period.

## Supported Versions

Security updates are provided for the latest version only. Users are encouraged to keep their installations up-to-date.

## Security Best Practices & Architecture

When using this application:

1.  **Frontend API Key:** The Google Cloud API key entered in the GUI is **only stored in your browser's local storage** and used for an initial validation call to the backend. It is **not** used by the backend for running agents. Treat it like sensitive data, but understand its limited use in this app.
2.  **Backend Credentials (Critical):** The Flask backend service requires Google Cloud credentials (ideally a **Service Account Key** specified via `GOOGLE_APPLICATION_CREDENTIALS` in the `.env` file or environment) to interact with Google AI services via the ADK. **These credentials are the most sensitive part.**
    *   **Protect your Service Account Key file.** Do not commit it to version control.
    *   Grant the service account the **least privilege** necessary to run the agents and tools you configure (e.g., Vertex AI User, BigQuery Read, etc.).
    *   Secure the environment where the backend service runs.
3.  **Agent Instructions & Tools:** Be cautious about instructions given to agents, especially if they use tools like `built_in_code_execution` or custom tools that interact with sensitive systems. Review generated code and tool logic.
4.  **Dependencies:** Regularly update frontend (`npm update`) and backend (`pip install --upgrade -r requirements.txt`) dependencies to patch vulnerabilities.
5.  **Deployment:** Secure your backend deployment (HTTPS, firewall rules, secure credential management). See `docs/deployment.md`.

## Third-Party Dependencies

This project uses third-party libraries (React, MUI, Flask, Google ADK, etc.). We strive to keep them updated. Vulnerabilities found in dependencies should also be reported via the process above, and we will update the affected library as soon as a fix is available.