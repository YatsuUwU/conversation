# AI-Powered Project Checklist and Assessment System

## Overview
This project provides an AI-powered system to help define, verify, and assess project readiness through a structured checklist approach. It leverages two AI models—Ollama and Google's Gemini—to generate checklist criteria, validate them with multiprompt questions, simulate a debate between optimistic and critical AI personas, and produce a final recommendation on whether to proceed with or reconsider the project.

## Main Script: `main.py`
The core functionality is implemented in `main.py`, which performs the following steps:

1. **Define Checklist Criteria**  
   Prompts the AI models to generate a list of 3-5 key checkpoints relevant to the specified industry or project type.  
   Example categories include technical feasibility, financial viability, regulatory compliance, resource availability, and market demand.

2. **Multiprompt Verification & Data Analysis**  
   For each checklist item, generates 2-3 specific validation questions. The user answers these questions interactively.  
   The AI then analyzes the answers to provide a brief assessment for each checkpoint and an overall project status summary.

3. **Counterargument Simulation**  
   Simulates a debate between two AI personas:  
   - Persona A (Pro-Project, optimistic) argues why the project should proceed.  
   - Persona B (Against-Project, critical) highlights risks and reasons for caution.

4. **AI-Generated Conclusion**  
   Based on the checklist verification, analysis, and debate, the AI provides a final recommendation to either proceed with or reconsider the project, including suggested next steps or modifications.

5. **Output**  
   The full project data, including checklist, verification results, debate arguments, and final conclusion, is saved to `project_assessment_output.json`.

## Configuration

- **Environment Variables**  
  - `GEMINI_API_KEY`: Required API key for Google's Gemini AI model. Must be set in the environment before running the script.

- **AI Models**  
  - **Ollama**: Optional local AI server at `http://localhost:11434` running the `llama2:latest` model. If unavailable, the system falls back to Gemini.  
  - **Gemini**: Google's generative AI model used as fallback and for certain tasks.

## Dependencies

- Python 3.x
- `requests` library for HTTP requests
- `google.generativeai` Python client for Gemini API
- Standard libraries: `os`, `json`, `sys`, `time`, `logging`

## Usage

1. Ensure the `GEMINI_API_KEY` environment variable is set.  
2. Optionally, run an Ollama server locally for enhanced performance.  
3. Run the script:  
   ```bash
   python main.py
   ```  
4. Follow the interactive prompts to enter the industry and answer validation questions.  
5. Review the printed AI analysis, debate, and final recommendation.  
6. Check the saved `project_assessment_output.json` for full structured output.

## Example Output Snippet

```json
{
  "industry": "software project",
  "checklist_criteria": [
    "Technical feasibility assessment",
    "Financial viability analysis",
    "Regulatory compliance check",
    "Resource availability confirmation",
    "Market demand validation"
  ],
  "verification_results": { ... },
  "gemini_analysis_report": "...",
  "debate": {
    "pro_argument_ollama": "...",
    "against_argument_gemini": "..."
  },
  "final_conclusion": "CONCLUSION: PROCEED ..."
}
```

## Logging

The script uses Python's `logging` module to provide info, warning, and error messages during execution, aiding in debugging and monitoring.

---

This system is designed as a prototype to assist project managers and stakeholders in systematically evaluating project readiness using AI-generated insights.
