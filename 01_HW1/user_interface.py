import streamlit as st
import os
import time
from google import genai
from google.genai.types import GenerateContentConfig
from dotenv import load_dotenv
 
# Load environment variables
load_dotenv()
 
# Set up Gemini API client
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in environment variables. Please set it in .env file.")
    st.stop()
 
client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL = "gemini-2.0-flash"
 
# Function to get Gemini response with rate limiting
def get_gemini_response(prompt):
    # Check last API call time for rate limiting
    if "last_call_time" in st.session_state:
        elapsed = time.time() - st.session_state.last_call_time
        if elapsed < 4:  # 4 seconds to respect 15 requests per minute
            time.sleep(4 - elapsed)
 
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.000001,
                max_output_tokens=512
            )
        )
        response_text = response.text.strip()
        
        # Update last call time
        st.session_state.last_call_time = time.time()
        return response_text
    except Exception as e:
        return f"Error querying Gemini API: {str(e)}"
 
# Streamlit UI
st.title("Gemini 2.0 Flash Prompt Evaluator")
 
# Input field
st.subheader("Enter Your Prompt")
prompt = st.text_area("Prompt", placeholder="Enter your prompt here...", height=150)
 
# Button to trigger API call
if st.button("Get Response"):
    if not prompt.strip():
        st.error("Please enter a prompt.")
    else:
        with st.spinner("Querying Gemini API..."):
            result = get_gemini_response(prompt)
        st.subheader("Gemini Response")
        st.text_area("Output", value=result, height=200, disabled=True)
 
# Instructions
st.markdown("""
### Instructions
1. Enter a prompt in the text box above.
2. Click the "Get Response" button to send the prompt to Gemini 2.0 Flash.
3. The response will appear in the output box below.
""")
 
