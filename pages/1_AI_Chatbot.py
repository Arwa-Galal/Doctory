import streamlit as st
from utils import MODELS # Import global MODELS dictionary
import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud.exceptions import NotFound

if MODELS is None: st.stop()

# --- Page Logic ---
st.header("AI Medical Assistant (MedBot) 💬")
st.markdown("Connect to your custom fine-tuned model (Gemma/Gemini) for conversational medical information.")

# --- Configuration Section for the LLM ---
with st.expander("⚙️ Configure LLM Connection", expanded=False):
    st.caption("These credentials link to your custom model hosted on Google Cloud Vertex AI. Requires prior authentication (`gcloud auth...`).")
    
    PROJECT_ID = st.text_input("GCP Project ID", value="gen-lang-client-0728569688")
    REGION = st.text_input("GCP Region", value="https://console.cloud.google.com/vertex-ai/dashboard?cloudshell=true&project=gen-lang-client-0728569688")
    TUNED_MODEL_NAME = st.text_input("Tuned Model Name (Full Resource Path)", value="models\medical_chatbot_notebook_py.ipynb")
    
    st.session_state['llm_config'] = {
        'PROJECT_ID': PROJECT_ID,
        'REGION': REGION,
        'TUNED_MODEL_NAME': TUNED_MODEL_NAME
    }

    if st.button("Initialize MedBot"):
        # Clear state to force re-initialization
        if "tuned_model" in st.session_state: del st.session_state["tuned_model"]
        if "messages" in st.session_state: del st.session_state["messages"]
        st.rerun()

# --- System Prompt Definition (guides the model's behavior) ---
medical_system_prompt = """
You are "MedBot," an AI assistant designed to provide helpful, general-purpose medical information. 
Your persona is professional, empathetic, and clear.

**Your core responsibilities are:**
1.  **Answer Clearly:** Provide accurate, easy-to-understand explanations for medical questions based on your specialized training data.
2.  **Be Informative, Not Diagnostic:** You can explain what conditions are, what symptoms are, and describe general treatment options. You MUST NOT diagnose, provide treatment plans, or interpret personal medical results.
3.  **Prioritize Safety:** If a user's query sounds like a medical emergency (e.g., "chest pain," "difficulty breathing," "severe bleeding"), your *first and only* response should be to advise them to seek immediate emergency medical help.

**CRITICAL SAFETY RULE:**
You MUST conclude every single response (except for emergency deflections) with the following disclaimer, formatted exactly like this:

---
*Disclaimer: I am an AI assistant and not a medical professional. This information is for educational purposes only. Please consult a qualified healthcare provider for medical advice, diagnosis, or treatment.*
"""

# --- Initialize Model and Chat History ---

if "tuned_model" not in st.session_state:
    st.session_state["tuned_model"] = None
    
    config = st.session_state.get('llm_config', {})
    if config and config['TUNED_MODEL_NAME'] and config['TUNED_MODEL_NAME'] != "projects/.../models/...":
        with st.spinner("Attempting to load custom model from Vertex AI..."):
            try:
                # 1. Initialize Vertex AI environment
                vertexai.init(project=config['PROJECT_ID'], location=config['REGION'])
                
                # 2. Load the specific model with the system instruction
                tuned_model = GenerativeModel(
                    config['TUNED_MODEL_NAME'],
                    system_instruction=[medical_system_prompt]
                )
                st.session_state["tuned_model"] = tuned_model
                st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I am MedBot, your specialized medical assistant. How can I help you today?"}]
                st.success("MedBot initialized successfully!")
            
            except NotFound:
                st.error("Error: Model not found. Check the Model Name and Project/Region.")
            except Exception as e:
                st.error(f"Error connecting to Vertex AI: {e}. Please ensure you are authenticated (`gcloud auth...`).")
    else:
         st.info("Please enter your GCP details and click 'Initialize MedBot' to start the chat.")


# --- Display Chat History ---

if "messages" in st.session_state:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Handle User Input ---

if st.session_state["tuned_model"] is not None:
    if prompt := st.chat_input("Ask a medical question..."):
        
        # Add user message to chat history
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from the LLM
        with st.chat_message("assistant"):
            with st.spinner("MedBot is thinking..."):
                try:
                    # Generate content using the loaded model
                    response = st.session_state["tuned_model"].generate_content([prompt])
                    
                    # Display the response
                    st.markdown(response.text)
                    
                    # Add assistant response to chat history
                    st.session_state["messages"].append({"role": "assistant", "content": response.text})
                    
                except Exception as e:
                    error_message = f"An error occurred during generation: {e}"
                    st.error(error_message)
                    st.session_state["messages"].append({"role": "assistant", "content": error_message})
