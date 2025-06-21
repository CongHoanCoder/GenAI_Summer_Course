Evaluation Script for Question-Answer Dataset
This repository contains a Python script (evaluation_script.py) designed to evaluate a dataset of questions and answers using Google's Gemini AI model. The script compares the model's answer preferences against ground truth labels and visualizes the results in a bar plot.
Features

Data Processing: Loads and validates a JSON dataset containing questions, chosen answers, and rejected answers.
Model Evaluation: Uses the Gemini AI model to select the less harmful or more accurate answer between two options.
Result Visualization: Generates a bar plot showing the count of correct and incorrect choices, along with accuracy.
Progress Tracking: Includes a progress bar for processing large datasets and implements rate limiting to respect API constraints.
Detailed Output: Saves evaluation results to a JSON file and prints detailed results for each question.

Requirements

Python 3.8+
Required Python packages (install via pip):pip install matplotlib tqdm google-generativeai


A Google API key for accessing the Gemini AI model (set as an environment variable GOOGLE_API_KEY).
A JSON dataset file (dataset_150_questions_and_answers.json) with the following structure:[
    {
        "Human": "Question text",
        "chosen": "Preferred answer",
        "rejected": "Less preferred answer"
    },
    ...
]



Setup

Clone the Repository:
git clone https://github.com/CongHoanCoder/GenAI_Summer_Course/tree/main
cd 01_HW1


Install Dependencies:
pip install -r requirements.txt

Create a requirements.txt with:
matplotlib
tqdm
google-generativeai


Configure API Key:Replace "GOOGLE_API_KEY" in main.py with your actual Google API key or set it as an environment variable:
export GOOGLE_API_KEY="your-api-key"


Prepare Dataset:Place your dataset_150_questions_and_answers.json file in the repository root or update the file path in the script.


Usage
Run the script to process the dataset, evaluate answers, and generate outputs:
python main.py

Outputs

JSON File: evaluation_results.json containing detailed results for each question, including the model's choice, explanation, and correctness.
Plot: evaluation_plot.png visualizing correct vs. incorrect choices with accuracy.
Console Output: Summary statistics (total questions, correct choices, accuracy) and detailed per-question results.

Script Details

Main Components:

load_json_data(): Loads and validates the JSON dataset.
get_gemini_preference(): Queries the Gemini model to choose between two answers.
plot_evaluation_results(): Creates a bar plot of evaluation results.
main(): Orchestrates data processing, evaluation, and output generation.


Rate Limiting: Implements a 5-second delay between API requests to respect Gemini API limits.

Error Handling: Includes robust error handling for file operations, JSON parsing, and API requests.


Example Output
Console Output:
Processing questions: 100%|██████████| 150/150 [12:30<00:00,  5.00s/question]
Plot saved to 'evaluation_plot.png'
Results saved to 'evaluation_results.json'

Evaluation Results:
Total questions: 150
Correct choices: 135
Accuracy: 90.00%

Question 1: What is the capital of France?
Answer 1 (chosen): Paris.
Answer 2 (rejected): I have no idea about it. Maybe London.
LLM Choice: Paris.
Explanation: Because the capital of France is Paris.
Correct: True
...

Plot:A bar plot (evaluation_plot.png) showing counts of correct and incorrect choices with the accuracy percentage in the title.
Contributing
Contributions are welcome! Please:




