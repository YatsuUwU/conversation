# Project Requirements

## Software Requirements

- **Python**: Version 3.x is required to run the project.
- **Python Packages**:
  - `requests` — for making HTTP requests to the Ollama server.
  - `google.generativeai` — Google Gemini API client for AI interactions.
- **AI Models**:
  - **Ollama** (optional): A local Ollama server running at `http://localhost:11434` with the `llama2:latest` model. If not available, the system falls back to Gemini.
  - **Gemini**: Google's generative AI model accessed via API.

## Environment Variables

- `GEMINI_API_KEY`: Required. Your API key for accessing the Google Gemini AI model. Must be set in your environment before running the script.

## Hardware Requirements

- No specific hardware requirements beyond what is needed to run Python and the AI models.
- Running the Ollama server locally may require additional resources depending on the model size.

## Setup Instructions

1. Install Python 3.x from [python.org](https://www.python.org/downloads/).
2. Install required Python packages:
   ```bash
   pip install requests google-generativeai
   ```
3. Set the `GEMINI_API_KEY` environment variable:
   - On Windows (Command Prompt):
     ```cmd
     set GEMINI_API_KEY=your_api_key_here
     ```
   - On Linux/macOS (bash):
     ```bash
     export GEMINI_API_KEY=your_api_key_here
     ```
4. (Optional) Start the Ollama server locally if you want to use it:
   - Follow Ollama's installation and setup instructions.
   - Ensure it is running at `http://localhost:11434`.

## Running the Project

Run the main script:
```bash
python main.py
```

Follow the interactive prompts to complete the project checklist assessment.

## Notes

- The project uses logging to provide runtime information and error messages.
- The output of the assessment is saved to `project_assessment_output.json`.
