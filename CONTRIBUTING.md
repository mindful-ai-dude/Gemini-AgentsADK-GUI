# Contributing to Gemini Agent Builder GUI

Thank you for considering contributing to the Gemini Agent Builder GUI! This document provides guidelines and instructions for contributing to this project.

**IMPORTANT**: This project is created by Gregory Kennedy (https://aiforsocialbenefit.xyz | peacedude@gmail.com)| peacedude@gmail.com) and is **not affiliated with, endorsed by, or sponsored by Google**. It is an independent tool designed to facilitate working with the **Google Agent Development Kit (ADK)**.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful, inclusive, and considerate in all interactions.

## How Can I Contribute?

### Reporting Bugs

If you find a bug in the GUI or the backend interaction, please create an issue with:
- A clear, descriptive title (e.g., `[BUG] Backend error when running agent with X tool`)
- Steps to reproduce the behavior (including agent configuration details if relevant).
- Expected behavior.
- Actual behavior (including error messages from the GUI and **backend terminal logs**).
- Screenshots (if applicable).
- Environment details (OS, Browser, Node.js version, Python version, ADK version).

### Suggesting Enhancements

We welcome suggestions! Please create an issue with:
- A clear, descriptive title (e.g., `[FEATURE] Add support for configuring X ADK parameter`).
- A detailed description of the proposed enhancement.
- Any relevant examples, mockups, or references to ADK documentation.
- Why this enhancement would be useful.

### Pull Requests

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/my-new-feature`).
3.  Make your changes to the frontend (React code in `src/`) and/or backend (Python code in `backend/`) as needed.
4.  Run linters/formatters if configured.
5.  Test your changes thoroughly (run both frontend and backend).
6.  Commit your changes (`git commit -m 'feat: Add support for X feature'`).
7.  Push to the branch (`git push origin feature/my-new-feature`).
8.  Open a Pull Request against the `main` branch.

## Development Setup

Refer to the main `README.md` for detailed installation and running instructions for both the frontend and backend.

## Coding Guidelines

### Frontend (JavaScript/React)
- Follow the existing code style (primarily based on Create React App standards).
- Use functional components with hooks.
- Add comments for complex logic.
- Keep components focused.
- Use Material UI components and styling.
- Ensure responsive design.

### Backend (Python/Flask)
- Follow PEP 8 guidelines.
- Use type hints.
- Add comments and docstrings for clarity.
- Keep Flask routes focused; separate business logic into helper functions or classes.
- Handle errors gracefully and provide informative JSON responses.
- Be mindful of security when handling configurations and interacting with Google Cloud APIs.

### Testing
- Add tests for new backend logic if possible (e.g., using `pytest`).
- Add tests for new complex frontend components/logic if possible (e.g., using React Testing Library).
- Ensure the application runs correctly end-to-end after your changes.

## Documentation
- Update the `README.md` or files in `docs/` if you change functionality, add features, or modify the setup/deployment process.
- Document new components, functions, or API endpoints.

## Review Process
1.  A maintainer will review your PR.
2.  Address any requested changes or improvements.
3.  Once approved, your PR will be merged.

## Thank You!
Your contributions help make this tool better for everyone working with Google ADK!