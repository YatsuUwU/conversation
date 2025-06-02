import os
import json
import requests
import google.generativeai as genai
import sys
import time
import logging

# Additional imports for new APIs
import cohere
from typing import Optional

# --- Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_CONFIG = "dolphin-mistral:7b" # Renamed to avoid conflict with the variable that can be set to None
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
HUGGINGFACE_API_TOKEN = os.environ.get('HUGGINGFACE_API_TOKEN')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
COHERE_API_KEY = os.environ.get('COHERE_API_KEY')

# --- Global variable for Ollama model, can be set to None if server is down ---
# This allows us to disable Ollama dynamically
OLLAMA_MODEL = OLLAMA_MODEL_CONFIG

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if not GEMINI_API_KEY:
    logger.critical("🚨 GEMINI_API_KEY environment variable not set. Please set it to proceed.")
    exit()

genai.configure(api_key=GEMINI_API_KEY)
gemini_model_instance = genai.GenerativeModel('gemini-1.5-flash-latest') # Renamed for clarity

# Setup Cohere client if API key is available
cohere_client: Optional[cohere.Client] = None
if COHERE_API_KEY:
    try:
        cohere_client = cohere.Client(COHERE_API_KEY)
        logger.info("Cohere client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Cohere client: {e}")

def query_huggingface(prompt, model_id="gpt2", timeout=30):
    """
    Query Hugging Face Inference API for text generation.
    Requires HUGGINGFACE_API_TOKEN environment variable.
    """
    if not HUGGINGFACE_API_TOKEN:
        logger.warning("HUGGINGFACE_API_TOKEN not set. Skipping Hugging Face query.")
        return None

    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
        "Accept": "application/json"
    }
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    payload = {
        "inputs": prompt,
        "options": {"wait_for_model": True}
    }
    try:
        logger.info(f"Querying Hugging Face model: {model_id}")
        response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "error" in data:
            logger.error(f"Hugging Face API error: {data['error']}")
            return None
        # The response is usually a list of generated texts
        if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        logger.error(f"Unexpected Hugging Face response format: {data}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Hugging Face API request failed: {e}")
        return None

def query_groq(prompt, system_message=None, model="llama3-8b-8192"):
    """
    Query Groq API for text generation.
    Requires GROQ_API_KEY environment variable.
    """
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set. Skipping Groq query.")
        return None
    try:
        from groq import Groq
    except ImportError:
        logger.error("Groq package not installed. Please install with 'pip install groq'.")
        return None

    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=model,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq API request failed: {e}")
        return None


project_data = {}

def query_gemini(prompt, model_instance=None, timeout=60):
    """
    Query Google Gemini API for text generation.
    Requires GEMINI_API_KEY environment variable.
    """
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set. Skipping Gemini query.")
        return None
    if model_instance is None:
        model_instance = gemini_model_instance

    try:
        logger.info("Querying Gemini model...")
        response = model_instance.generate_content(prompt, request_options={'timeout': timeout})
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini API request failed: {e}")
        return None

def query_cohere(prompt, model="command-r"):
    """
    Query Cohere API for text generation.
    Requires COHERE_API_KEY environment variable.
    """
    if not cohere_client:
        logger.warning("Cohere client not initialized. Skipping Cohere query.")
        return None
    try:
        response = cohere_client.chat(
            model=model,
            message=prompt
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Cohere API request failed: {e}")
        return None

project_data = {}

def step_1_define_criteria():
    logger.info("\n--- 1️⃣ Define the Checklist Criteria ---")
    industry = input("Please enter the industry for the software project (e.g., 'FinTech', 'Healthcare', 'E-commerce'): ")
    project_data['industry'] = industry

    prompt = f"""
    You are an AI assistant helping to define project readiness checklists.
    For a project in the '{industry}' industry/domain, generate a structured list of 3-5 key checkpoints
    that must be met before proceeding.
    Focus on high-level categories like: financial feasibility, regulatory compliance, technical readiness.

    Example for 'software project': Scalability, security compliance, API integrations.
    Example for 'healthcare AI solution': FDA approval, patient data protection, ethical AI use.

    Output the list as a JSON array of strings. For example:
    ["Checkpoint 1", "Checkpoint 2", "Checkpoint 3"]
    """

    raw_checklist = None
    if OLLAMA_MODEL: # Check if Ollama is enabled
        logger.info(f"🤖 Asking Ollama ({OLLAMA_MODEL}) to generate checklist criteria...")
        raw_checklist = query_ollama(prompt)

    if raw_checklist is None:
        logger.warning("⚠️ Ollama failed or was skipped. Falling back to Gemini for checklist generation...")
        raw_checklist = query_gemini(prompt)

    if raw_checklist is None:
        logger.warning("⚠️ Gemini failed or was skipped. Falling back to Hugging Face for checklist generation...")
        raw_checklist = query_huggingface(prompt, model_id="gpt2")

    if raw_checklist is None:
        logger.warning("⚠️ Hugging Face failed or was skipped. Falling back to Groq for checklist generation...")
        raw_checklist = query_groq(prompt)

    if raw_checklist is None:
        logger.warning("⚠️ Groq failed or was skipped. Falling back to Cohere for checklist generation...")
        raw_checklist = query_cohere(prompt)

    logger.info(f"AI's raw response for checklist:\n{raw_checklist}")

    try:
        if raw_checklist is None:
            raise ValueError("AI failed to provide any checklist response.")
        parsed_checklist = json.loads(raw_checklist)
        if not isinstance(parsed_checklist, list) or not all(isinstance(item, str) for item in parsed_checklist):
            raise ValueError("Checklist is not a list of strings.")
        project_data['checklist_criteria'] = parsed_checklist
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"⚠️ AI did not return a valid JSON list for checklist. Error: {e}")
        logger.info("Attempting to extract checklist items manually or generate new using Gemini...")
        cleanup_prompt = f"""
        The following text is supposed to be a JSON list of project checklist items, but it might be malformed or missing:
        ---
        {raw_checklist if raw_checklist else "No content provided by previous AI."}
        ---
        Please extract the checklist items and format them as a valid JSON array of strings.
        If you cannot extract a meaningful list, generate a new list of 3-5 checklist items for a '{industry}' project.
        """
        cleaned_checklist_str = query_gemini(cleanup_prompt)
        logger.info(f"Gemini's cleaned or generated checklist string: {cleaned_checklist_str}")
        try:
            if cleaned_checklist_str is None:
                raise ValueError("Gemini failed to provide a cleaned/new checklist.")
            project_data['checklist_criteria'] = json.loads(cleaned_checklist_str)
            if not isinstance(project_data['checklist_criteria'], list) or len(project_data['checklist_criteria']) == 0:
                raise ValueError("Invalid or empty checklist from Gemini cleanup.")
        except (json.JSONDecodeError, ValueError) as e_cleanup:
            logger.error(f"🚨 Failed to produce a valid JSON checklist even after cleanup. Error: {e_cleanup}. Generating a default list.")
            project_data['checklist_criteria'] = [
                "Technical feasibility assessment",
                "Financial viability analysis",
                "Regulatory compliance check",
                "Resource availability confirmation",
                "Market demand validation"
            ]

    logger.info("\n✅ Generated Checklist Criteria:")
    for i, item in enumerate(project_data.get('checklist_criteria', [])):
        print(f"   {i+1}. {item}") # Using print for direct user output


def step_2_multiprompt_verification():
    logger.info("\n--- 2️⃣ Multiprompt Verification & Data Analysis ---")
    if not project_data.get('checklist_criteria') or \
       ("Error" in project_data['checklist_criteria'][0] if project_data['checklist_criteria'] else True):
        logger.error("🚨 Cannot proceed: Checklist criteria not defined or error in definition.")
        project_data['verification_results'] = {}
        project_data['gemini_analysis_report'] = "Skipped: Checklist definition failed."
        return

    project_data['verification_results'] = {}
    user_answers_summary = []

    for i, item in enumerate(project_data['checklist_criteria']):
        print(f"\n🔍 Verifying Checkpoint {i+1}: {item}") # User-facing print

        prompt_questions = f"""
        For the project checklist item: '{item}' in a '{project_data['industry']}' context,
        generate 2-3 specific validation questions to assess if this checkpoint is met.
        Frame them as direct questions the user should answer.
        Output as a JSON array of strings. Example:
        ["Question 1?", "Question 2?"]
        """
        
        raw_questions = None
        if OLLAMA_MODEL:
            logger.info(f"🤖 Asking Ollama ({OLLAMA_MODEL}) for validation questions...")
            raw_questions = query_ollama(prompt_questions)
        
        if raw_questions is None:
            logger.warning("⚠️ Ollama failed or was skipped for question generation. Falling back to Gemini...")
            raw_questions = query_gemini(prompt_questions)

        if raw_questions is None:
            logger.warning("⚠️ Gemini failed or was skipped for question generation. Falling back to Hugging Face...")
            raw_questions = query_huggingface(prompt_questions, model_id="gpt2")

        if raw_questions is None:
            logger.warning("⚠️ Hugging Face failed or was skipped for question generation. Falling back to Groq...")
            raw_questions = query_groq(prompt_questions)

        if raw_questions is None:
            logger.warning("⚠️ Groq failed or was skipped for question generation. Falling back to Cohere...")
            raw_questions = query_cohere(prompt_questions)

        logger.info(f"AI's raw response for questions:\n{raw_questions}")

        questions = None
        try:
            if raw_questions is None:
                raise ValueError("AI failed to provide questions.")
            questions = json.loads(raw_questions)
            if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
                raise ValueError("Not a list of strings")
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"⚠️ AI did not return valid JSON questions. Error: {e}. Using generic questions.")
            questions = [f"What is the status of '{item}'?", f"What evidence supports the completion of '{item}'?"]

        item_answers = {"checkpoint": item, "questions": questions, "answers": []}
        for q_idx, q_text in enumerate(questions):
            answer = input(f"    Q{q_idx+1}: {q_text}\n   Your Answer: ")
            item_answers["answers"].append({"question": q_text, "answer": answer})
            user_answers_summary.append(f"For '{item}', Q: {q_text} A: {answer}")
            time.sleep(0.2) # Reduced sleep time

        project_data['verification_results'][item] = item_answers

    user_answers_full_text = "\n".join(user_answers_summary)
    prompt_analysis = f"""
    Project Type: {project_data['industry']}
    Checklist Criteria and User Answers:
    {user_answers_full_text}

    Based on the user's answers:
    1. Analyze the provided information for each checkpoint.
    2. Briefly "cross-reference" against common knowledge of outcomes or best practices for such projects.
       (e.g., "The financial projections seem optimistic given typical market entry costs for similar SaaS products.")
    3. Provide a short (1-2 sentence) assessment for each checkpoint (e.g., "Seems well-addressed", "Requires more detail", "Potential risk area").
    4. Provide an overall brief summary statement about the project's current state based on these answers.

    Output your analysis clearly, perhaps point by point for each checkpoint, and then the overall summary.
    """
    logger.info("\n🤖 Asking Gemini to analyze answers and simulate cross-referencing...")
    analysis_report = query_gemini(prompt_analysis)
    project_data['gemini_analysis_report'] = analysis_report if analysis_report else "Gemini failed to provide analysis."
    
    print("\n📊 Gemini's Analysis Report:") # User-facing print
    print(project_data['gemini_analysis_report'])


def step_3_counterargument_simulation():
    logger.info("\n--- 3️⃣ Counterargument Simulation – Debate Between Two AI Personas ---")
    if not project_data.get('checklist_criteria') or \
       not project_data.get('gemini_analysis_report') or \
       "Skipped" in project_data.get('gemini_analysis_report', ""):
        logger.warning("🚨 Cannot proceed with debate: Checklist or analysis report missing/skipped.")
        project_data['debate'] = {"pro_argument_ollama": "Skipped", "against_argument_gemini": "Skipped due to missing prior data."}
        return

    project_summary = f"""
    Project Type: {project_data['industry']}
    Current Status & Analysis:
    {json.dumps(project_data.get('verification_results'), indent=2)}
    Gemini's Initial Analysis: {project_data.get('gemini_analysis_report')}
    """

    pro_argument = None
    if OLLAMA_MODEL:
        prompt_pro = f"""
        You are Persona A, an optimistic but realistic project advocate.
        Given the following project summary, argue WHY the project should proceed.
        Address potential risks highlighted in the analysis by offering potential solutions or mitigations.
        Keep your argument concise (2-3 key points).

        Project Summary:
        {project_summary}

        Your Pro-Project Argument:
        """
        logger.info(f"\n🤖 Asking Ollama ({OLLAMA_MODEL}) for Pro-Project Argument (Persona A)...")
        pro_argument = query_ollama(prompt_pro)
    
    if pro_argument is None and OLLAMA_MODEL: # Log if Ollama was attempted but failed
        logger.warning("Ollama failed to provide a pro-argument.")
    elif pro_argument is None: # Ollama was skipped
         logger.info("Ollama was skipped for pro-argument. Pro-argument will be missing or generated by Gemini if implemented as fallback.")
         # For now, we just note it's missing if Ollama is the designated pro-arguer
         pro_argument = "Pro-argument (Ollama) not available or failed."


    project_data['debate_pro_argument'] = pro_argument
    print("\n👍 Persona A (Pro-Project - Ollama):") # User-facing print
    print(pro_argument)

    prompt_against = f"""
    You are Persona B, a cautious and critical project evaluator.
    Given the following project summary AND the Pro-Project argument (if available), highlight critical challenges,
    unaddressed risks, or reasons why the project might fail or needs significant reconsideration.
    Recommend caution if necessary. Keep your argument concise (2-3 key points).

    Project Summary:
    {project_summary}

    Persona A's Pro-Project Argument:
    {pro_argument if pro_argument else "Pro-argument not available."}

    Your Critical Counter-Argument (Persona B):
    """
    logger.info("\n🤖 Asking Gemini for Against-Project Argument (Persona B)...")
    against_argument = query_gemini(prompt_against)
    project_data['debate_against_argument'] = against_argument if against_argument else "Gemini failed to provide counter-argument."
    
    print("\n👎 Persona B (Against-Project - Gemini):") # User-facing print
    print(project_data['debate_against_argument'])

    project_data['debate'] = {
        "pro_argument_ollama": project_data['debate_pro_argument'],
        "against_argument_gemini": project_data['debate_against_argument']
    }

def step_4_ai_generated_conclusion():
    logger.info("\n--- 4️⃣ AI-Generated Conclusion – Proceed or Reconsider? ---")
    if not project_data.get('debate') or \
       "Skipped" in project_data['debate'].get('pro_argument_ollama', "") or \
       "Skipped" in project_data['debate'].get('against_argument_gemini', ""):
        logger.warning("🚨 Cannot proceed with final conclusion: Debate simulation missing or skipped.")
        project_data['final_conclusion'] = "Reconsider: Debate simulation was not performed or failed."
        return

    # Ensure debate arguments are strings for the prompt
    pro_arg_text = project_data['debate'].get('pro_argument_ollama', "Not available")
    if pro_arg_text is None: pro_arg_text = "Not available"
    
    against_arg_text = project_data['debate'].get('against_argument_gemini', "Not available")
    if against_arg_text is None: against_arg_text = "Not available"

    conclusion_input = f"""
    Project Type: {project_data['industry']}
    Checklist Verification Summary: {json.dumps(project_data.get('verification_results'), indent=2)}
    Initial Gemini Analysis: {project_data.get('gemini_analysis_report', 'Not available')}
    Debate:
      Pro-Project (Ollama): {pro_arg_text}
      Against-Project (Gemini): {against_arg_text}

    Based on ALL the information above (checklist results, initial analysis, and the simulated debate),
    provide a final recommendation: Should the project PROCEED or RECONSIDER?

    If PROCEED:
      - Briefly confirm feasibility.
      - Suggest 1-2 high-level next execution steps.
    If RECONSIDER:
      - Clearly identify the 1-2 most critical flaws or risks.
      - Suggest specific modifications or alternative approaches.

    Be decisive but base your recommendation on the provided evidence.
    Start your response with either "CONCLUSION: PROCEED" or "CONCLUSION: RECONSIDER".
    """
    logger.info("\n🤖 Asking Gemini for the Final Conclusion...")
    final_conclusion = query_gemini(conclusion_input)
    project_data['final_conclusion'] = final_conclusion if final_conclusion else "Gemini failed to provide a final conclusion."
    
    print("\n🏁 AI's Final Recommendation (Gemini):") # User-facing print
    print(project_data['final_conclusion'])


def check_ollama_server():
    """
    Checks if the Ollama server is running and if the configured model is available.
    Sets the global OLLAMA_MODEL to None if the server is unreachable or model is not found.
    """
    global OLLAMA_MODEL
    logger.info(f"Attempting to connect to Ollama server at {OLLAMA_BASE_URL}...")
    try:
        # Check if server is reachable
        requests.get(OLLAMA_BASE_URL, timeout=5)
        logger.info("Ollama server is reachable.")

        # Check if the model is available
        model_list_url = f"{OLLAMA_BASE_URL}/api/tags"
        response = requests.get(model_list_url, timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        available_models = [m["name"] for m in models]

        if OLLAMA_MODEL_CONFIG not in available_models:
            logger.warning(f"🚨 Configured Ollama model '{OLLAMA_MODEL_CONFIG}' not found on server. Available models: {', '.join(available_models) if available_models else 'None'}")
            OLLAMA_MODEL = None
        else:
            logger.info(f"Configured Ollama model '{OLLAMA_MODEL_CONFIG}' is available.")

    except requests.exceptions.ConnectionError:
        logger.warning(f"🚨 Ollama server not found at {OLLAMA_BASE_URL}. Disabling Ollama functionality.")
        OLLAMA_MODEL = None
    except requests.exceptions.Timeout:
        logger.warning(f"🚨 Connection to Ollama server timed out at {OLLAMA_BASE_URL}. Disabling Ollama functionality.")
        OLLAMA_MODEL = None
    except requests.exceptions.RequestException as e:
        logger.warning(f"🚨 An error occurred while checking Ollama server: {e}. Disabling Ollama functionality.")
        OLLAMA_MODEL = None


def query_ollama(prompt, model=None, timeout=60):
    """
    Query the Ollama server for text generation. Returns the generated text or None on failure.
    """
    if OLLAMA_MODEL is None:
        logger.info("Ollama is disabled or unavailable. Skipping Ollama query.")
        return None
    if model is None:
        model = OLLAMA_MODEL
    api_url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        logger.info(f"Querying Ollama model: {model}")
        response = requests.post(api_url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if "response" in data:
            return data["response"].strip()
        logger.error(f"Unexpected Ollama response format: {data}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API request failed: {e}")
        return None


if __name__ == "__main__":
    print("🚀 AI-Powered Checklist System Prototype 🚀") # User-facing print

    check_ollama_server() # This will set OLLAMA_MODEL to None if server is down

    step_1_define_criteria()
    
    # Proceed only if checklist criteria were successfully defined
    if project_data.get('checklist_criteria') and \
       not ("Error" in project_data['checklist_criteria'][0] if project_data['checklist_criteria'] else True):
        step_2_multiprompt_verification()
        step_3_counterargument_simulation()
        step_4_ai_generated_conclusion()
    else:
        logger.error("\n🚨 Workflow halted due to issues in Step 1: Unable to generate valid checklist criteria.")
        # Ensure essential keys exist for JSON output even if steps are skipped
        if 'verification_results' not in project_data: project_data['verification_results'] = {}
        if 'gemini_analysis_report' not in project_data: project_data['gemini_analysis_report'] = "Skipped due to checklist failure."
        if 'debate' not in project_data: project_data['debate'] = {"pro_argument_ollama": "Skipped", "against_argument_gemini": "Skipped"}
        if 'final_conclusion' not in project_data: project_data['final_conclusion'] = "Reconsider: Workflow halted early."


    logger.info("\n--- 📝 Full Project Data Collected ---")
    try:
        with open("project_assessment_output.json", "w") as f:
            json.dump(project_data, f, indent=2)
        logger.info("\nFull project data saved to project_assessment_output.json")
    except Exception as e:
        logger.error(f"Failed to save project data to JSON: {e}")
        
    print("\n💡 Prototype workflow complete! 💡") # User-facing print
