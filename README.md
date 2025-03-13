# CSV Query Assistant

A Gradio-based application that allows users to upload CSV files, ask natural language questions about the data, and visualize the results using a local Large Language Model (LLM).

## Features

- Upload and analyze CSV files
- Ask questions in natural language about your data
- Get answers to both textual and numerical queries
- Generate visualizations based on your data
- Powered by a local LLM via Ollama and PydanticAI


## Setup Instructions

### Prerequisites
- Ensure you have Python and pip installed.
- Install required Python packages: `pip install gradio pandas matplotlib seaborn pillow pydantic-ai`.

### Ollama Setup
1. **Download and Install Ollama**: Follow the instructions on the Ollama website.

2. **Run the Model**:
   - Start the Ollama server with the desired model:
     `ollama run llama3.1`
### Usage

1. Start the application:
`python app_v3.py`

2. Open your browser and navigate to the URL displayed in the terminal
3. Upload a CSV file
4. Ask questions about your data in natural language
5. View the answers and visualizations

## Demo

Here are some examples of the application in action:

 ![Screenshot (943)](https://github.com/user-attachments/assets/26d1ca48-c3a2-422d-82d6-9eff13cb5ad0)


 ![Screenshot (942)](https://github.com/user-attachments/assets/d03b04f8-fba6-4067-9416-b0da1989015e)
