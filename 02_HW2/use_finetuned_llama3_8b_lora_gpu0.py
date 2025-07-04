import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from bert_score import score
from huggingface_hub import login
import os


#####################################################################################################
###
###          SETUP
###
#####################################################################################################

# Set GPU 0 explicitly
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Authenticate with Hugging Face
HF_TOKEN = ""  # Replace with your token
try:
    login(token=HF_TOKEN)
    print("Hugging Face authentication successful")
except Exception as e:
    print(f"Error authenticating with Hugging Face: {e}")
    exit(1)

# Load base model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
adapter_dir = "./llama3_8b_lora_adapters_gpu0"
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},  # GPU 0
        # load_in_4bit=True,  # 4-bit quantization
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    print(f"Error loading base model: {e}")
    exit(1)

# Load LoRA adapters
try:
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    print(f"Loaded LoRA adapters from {adapter_dir}")
except Exception as e:
    print(f"Error loading LoRA adapters: {e}")
    exit(1)

# Zero-shot prompt template
zero_shot_prompt = """Instruction: {instruction}
Output: """


#####################################################################################################
###
###          Generate Output
###
#####################################################################################################

# Example: Single-prompt inference
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
        print(f"Error generating output: {e}")
        return ""


#####################################################################################################
###
###          Test and Evaluate BERTScore
###
#####################################################################################################

# Test single prompt
sample_instruction = "Draft a legal memorandum summarizing the key issues in a contract dispute case."
print("Single-prompt example:")
print(f"Instruction: {sample_instruction}")
print(f"Output: {generate_single_output(sample_instruction)}\n")

# Load dataset for batch inference and BERTScore
input_file = "lawyer_instruct_300_examples_1001_1300.json"
try:
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if len(data) == 0:
        raise ValueError("JSON file is empty")
    print(f"Loaded {len(data)} examples from {input_file}")
except Exception as e:
    print(f"Error loading JSON: {e}")
    exit(1)

# Verify data structure
if not all("instruction" in ex and "output" in ex for ex in data):
    print("Error: Invalid data format. Expected 'instruction' and 'output' keys.")
    exit(1)

# Batch inference for 1000 examples
generated_outputs = []
references = [ex["output"] for ex in data]

for i, example in enumerate(data):
    instruction = example["instruction"]
    generated_output = generate_single_output(instruction)
    generated_outputs.append(generated_output)
    if i % 10 == 0:
        print(f"Processed {i} examples")

# Compute BERTScore
try:
    P, R, F1 = score(
        generated_outputs,
        references,
        lang="en",
        model_type="roberta-large",
        verbose=True
    )
    bert_scores = {
        "precision": P.tolist(),
        "recall": R.tolist(),
        "f1": F1.tolist(),
        "average_precision": float(P.mean()),
        "average_recall": float(R.mean()),
        "average_f1": float(F1.mean())
    }
except Exception as e:
    print(f"Error computing BERTScore: {e}")
    exit(1)

# Save BERTScore results
output_file = "llama3_8b_finetuned_bertscore_zero_shot_results_with_300_examples.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(bert_scores, f, indent=2)

print(f"Saved BERTScore results to {output_file}")
print(f"Average BERTScore - Precision: {bert_scores['average_precision']:.4f}, "
      f"Recall: {bert_scores['average_recall']:.4f}, "
      f"F1: {bert_scores['average_f1']:.4f}")