import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image
import asyncio
import time
import os
import signal
import sys
from dataclasses import dataclass
from typing import Any, Dict
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Define dependencies for the agent
@dataclass
class DataFrameDependencies:
    df: pd.DataFrame

class DataFrameQueryResult(BaseModel):
    answer: str = Field(description="The answer to the query about the dataframe")
    code_executed: str = Field(description="The pandas code that was executed to answer the query")

# Define Ollama as a provider
ollama_model = OpenAIModel(
    model_name="granite3.2-vision:2b",  # Match the model name running in Ollama
    provider=OpenAIProvider(base_url="http://localhost:11434/v1")  # Ollama's API URL
)

df_agent = Agent(
    ollama_model,
    deps_type=DataFrameDependencies,
    result_type=DataFrameQueryResult,
    system_prompt=(
        "You are a data analysis assistant that helps users analyze pandas DataFrames. "
        "Convert natural language queries into pandas Python code and execute it."
    )
)

# Tool to execute queries
@df_agent.tool
def execute_pandas_query(ctx: RunContext[DataFrameDependencies], code: str) -> Dict[str, Any]:
    """Executes pandas code on the dataframe."""
    df = ctx.deps.df
    try:
        local_vars = {"df": df, "pd": pd}
        result = eval(code, {"__builtins__": {}}, local_vars)

        if isinstance(result, pd.DataFrame):
            return {"type": "dataframe", "data": result.head(10).to_dict(orient='records')}
        elif isinstance(result, pd.Series):
            return {"type": "series", "data": result.head(10).to_dict()}
        else:
            return {"type": "other", "data": str(result)}
    except Exception as e:
        return {"error": str(e)}

# Function to process queries asynchronously
async def process_dataframe_query(df, query):
    # For large dataframes, use a sample to speed up processing
    if len(df) > 100000:
        sample_df = df.sample(100000)
        deps = DataFrameDependencies(df=sample_df)
    else:
        deps = DataFrameDependencies(df=df)
    
    try:
        # Add timeout to the agent run
        result = await asyncio.wait_for(
            df_agent.run(query, deps=deps),
            timeout=300  # 5 minutes timeout
        )
        return result.data
    except asyncio.TimeoutError:
        raise Exception("Processing took too long. Try a simpler question or smaller dataset.")

# Gradio function to handle CSV upload
def upload_file(file):
    try:
        df = pd.read_csv(file.name)
        return df.head().to_string()
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio function to answer user questions about the CSV
def answer_question(file, question, progress=gr.Progress()):
    try:
        df = pd.read_csv(file.name)
        
        # Initialize progress
        progress(0, desc="Loading data...")
        time.sleep(0.5)  # Simulate initial loading
        
        progress(0.2, desc="Processing query...")
        result = asyncio.run(process_dataframe_query(df, question))
        
        progress(1.0, desc="Complete!")
        return result
    except Exception as e:
        return f"Error processing your question: {str(e)}"

# Gradio function to plot a graph from a column
def plot_graph(file, column, progress=gr.Progress()):
    try:
        progress(0, desc="Loading data...")
        df = pd.read_csv(file.name)
        
        progress(0.3, desc="Analyzing data...")
        plt.figure(figsize=(8, 5))
        
        progress(0.6, desc="Creating visualization...")
        df[column].value_counts().plot(kind='bar')
        plt.title(f"Plot of {column}")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        progress(0.9, desc="Finalizing...")
        img = Image.open(buf)
        
        progress(1.0, desc="Complete!")
        return img
    except Exception as e:
        return f"Error creating plot: {str(e)}"

# Function to gracefully shutdown the server
def shutdown():
    print("Shutting down server...")
    time.sleep(0.5)
    os._exit(0)

# Create a custom theme for better appearance
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="purple",
)

# Gradio UI components with queue enabled
with gr.Blocks(theme=custom_theme, title="CSV Data Analysis Assistant") as app:
    gr.Markdown("# CSV Data Analysis Assistant")
    gr.Markdown("Upload a CSV file and analyze it using natural language queries or create visualizations.")
    
    with gr.Tab("Upload"):
        with gr.Row():
            file_input1 = gr.File(label="Upload CSV")
        with gr.Row():
            preview_button = gr.Button("Preview Data")
            preview_output = gr.Textbox(label="Data Preview")
        preview_button.click(fn=upload_file, inputs=file_input1, outputs=preview_output)
    
    with gr.Tab("Question Answering"):
        with gr.Row():
            file_input2 = gr.File(label="Upload CSV")
        with gr.Row():
            question_input = gr.Textbox(label="Ask a Question", placeholder="E.g., What is the average of column X?")
        with gr.Row():
            answer_button = gr.Button("Get Answer")
            answer_output = gr.Textbox(label="Answer")
        answer_button.click(fn=answer_question, inputs=[file_input2, question_input], outputs=answer_output)
    
    with gr.Tab("Plotting"):
        with gr.Row():
            file_input3 = gr.File(label="Upload CSV")
        with gr.Row():
            column_input = gr.Textbox(label="Column to Plot", placeholder="Enter column name")
        with gr.Row():
            plot_button = gr.Button("Generate Plot")
            plot_output = gr.Image(label="Generated Plot")
        plot_button.click(fn=plot_graph, inputs=[file_input3, column_input], outputs=plot_output)
    
    with gr.Row():
        shutdown_button = gr.Button("Shutdown Server", variant="stop")
    
    shutdown_button.click(fn=shutdown, inputs=None, outputs=None)

# Set up signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("Shutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Launch the app with proper queue settings
app.queue(max_size=20).launch(
    server_name="127.0.0.1",  # Use localhost instead of 0.0.0.0
    server_port=7860,
    share=False,
    max_threads=40
)
