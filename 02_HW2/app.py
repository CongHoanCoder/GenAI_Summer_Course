import streamlit as st
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login
import os

# Workaround for Streamlit torch.classes error
from streamlit.watcher import local_sources_watcher
def safe_get_module_paths(module):
    try:
        return list(module.__path__._path)
    except Exception:
        return []
local_sources_watcher.extract_paths = safe_get_module_paths

# Set GPU 0 explicitly
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Authenticate with Hugging Face
HF_TOKEN = ""
try:
    login(token=HF_TOKEN)
    st.success("Hugging Face authentication successful")
except Exception as e:
    st.error(f"Error authenticating with Hugging Face: {e}")
    st.error("Troubleshooting: Check HF token, internet (ping huggingface.co), or clear cache (rm -rf ~/.cache/huggingface)")
    st.stop()

# Load base model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
adapter_dir = "./llama3_8b_lora_adapters_gpu0"
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},  # GPU 0
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    st.error(f"Error loading base model: {e}")
    st.error("Troubleshooting: Check internet (ping huggingface.co), HF token, GPU 0 (nvidia-smi), or clear cache (rm -rf ~/.cache/huggingface)")
    st.stop()

# Load LoRA adapters
try:
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    st.success(f"Loaded LoRA adapters from {adapter_dir}")
except Exception as e:
    st.error(f"Error loading LoRA adapters: {e}")
    st.error("Troubleshooting: Verify ./llama3_8b_lora_adapters_gpu0 exists")
    st.stop()

# Zero-shot prompt template
zero_shot_prompt = """Instruction: {instruction}
Output: """

# Function to generate output
@st.cache_resource
def generate_single_output(instruction):
    prompt = zero_shot_prompt.format(instruction=instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text.split("Output:")[1].strip() if "Output:" in generated_text else generated_text
    except Exception as e:
        return f"Error generating output: {e}"

# Streamlit UI
st.title("Fine-Tuned Meta-Llama-3-8B Legal Reasoning Demo")
st.write("Enter a legal instruction to generate a response using the fine-tuned Meta-Llama-3-8B model.")

# Input field
instruction = st.text_area("Instruction", placeholder="e.g., Draft a legal memorandum summarizing the key issues in a contract dispute case.", height=100)

# Generate button
if st.button("Generate Response"):
    if not instruction:
        st.warning("Please enter an instruction.")
    else:
        with st.spinner("Generating response..."):
            output = generate_single_output(instruction)
            st.subheader("Generated Output")
            st.write(output)

# Instructions for use
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Enter a legal instruction in the text box (e.g., 'Draft a legal memorandum...').
2. Click 'Generate Response' to see the model's output.
3. Ensure GPU 0 is available (check with `nvidia-smi`).
4. Troubleshooting:
   - Verify HF token and internet.
   - Clear cache: `rm -rf ~/.cache/huggingface`.
   - Check ./llama3_8b_lora_adapters_gpu0 directory.
""")