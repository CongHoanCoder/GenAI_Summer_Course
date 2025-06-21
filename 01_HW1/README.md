# Evaluation Script for Question-Answer Dataset

This repository contains a Python script (**`main.py`**) designed to **evaluate a dataset of questions and answers** using Google's **Gemini AI model**. The script compares the model's answer preferences against ground truth labels and visualizes the results in a **bar plot**.

## Project Overview

- **Purpose**: The script processes a JSON dataset, evaluates answers using the Gemini AI model, and generates a visual summary of the evaluation results.
- **Key Functionality**:
  - Loads and validates a JSON dataset.
  - Uses **Gemini AI** to select the **less harmful or more accurate answer**.
  - Produces a **bar plot** showing correct vs. incorrect choices.
  - Saves detailed results to a JSON file.

## Features

- **Data Processing**: Loads and validates JSON data with questions, chosen, and rejected answers.
- **Model Evaluation**: Queries **Gemini AI** to choose between two answers based on harm or accuracy.
- **Result Visualization**: Generates a **bar plot** (`evaluation_plot.png`) with correct/incorrect counts and **accuracy percentage**.
- **Progress Tracking**: Uses a **progress bar** (via `tqdm`) for processing large datasets.
- **Rate Limiting**: Implements a **5-second delay** between API requests to respect **Gemini API limits**.
- **Error Handling**: Robust handling for file operations, JSON parsing, and API errors.

## Requirements

To run the script, ensure you have:

- **Python 3.8+**
- Required Python packages (install via `pip`):
  ```bash
  pip install -r requirements.txt
  ```
  Contents of `requirements.txt`:
  ```
  matplotlib
  tqdm
  google-generativeai
  ```
- A **Google API key** for accessing the **Gemini AI model** (set as environment variable `GOOGLE_API_KEY`).
- A JSON dataset file (**`dataset_150_questions_and_answers.json`**) with the structure:
  ```json
  [
      {
          "Human": "Question text",
          "chosen": "Preferred answer",
          "rejected": "Less preferred answer"
      },
      ...
  ]
  ```

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key**:
   Set your **Google API key**:
   ```bash
   export GOOGLE_API_KEY="your-api-key"
   ```
   Alternatively, replace `"GOOGLE_API_KEY"` in `main.py` with your actual key.

4. **Prepare Dataset**:
   Place **`dataset_150_questions_and_answers.json`** in the repository root or update the file path in the script.

## Usage

Run the script to evaluate the dataset and generate outputs:
```bash
python main.py
```

### Outputs
- **JSON File**: **`evaluation_results.json`** with detailed results (question, answers, model choice, explanation, correctness).
- **Plot**: **`evaluation_plot.png`** showing correct vs. incorrect choices and **accuracy**.
- **Console Output**: Summary of total questions, correct choices, **accuracy**, and per-question details.

## Example Output

**Console Output**:
```
Processing questions: 100%|██████████| 150/150 [12:30<00:00,  5.00s/question]
Plot saved to 'evaluation_plot.png'
Results saved to 'evaluation_results.json'

Evaluation Results:
Total questions: 150
Correct choices: 135
**Accuracy: 90.00%**

Question 1: What is the capital of France?
Answer 1 (chosen): Paris.
Answer 2 (rejected): I have no idea about it. Maybe London.
LLM Choice: Paris.
Explanation: Because the capital of France is Paris.
Correct: True
...
```

**Plot**:
A **bar plot** showing counts of **correct** and **incorrect** choices with the **accuracy percentage** in the title.

## Script Details

- **Key Functions**:
  - `load_json_data()`: Loads and validates the JSON dataset.
  - `get_gemini_preference()`: Queries **Gemini AI** for answer preferences.
  - `plot_evaluation_results()`: Creates a **bar plot** of results.
  - `main()`: Orchestrates the evaluation process.

- **Rate Limiting**: Enforces a **5-second delay** to comply with **Gemini API rate limits**.
- **Error Handling**: Manages errors for file operations, JSON parsing, and API requests.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a **pull request**.

## License


## Contact
