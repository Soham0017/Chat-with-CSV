import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from typing import Dict, Any, Optional
import numpy as np
import traceback
import asyncio
from dataclasses import dataclass
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Initialize the LLM agent with Ollama
ollama_model = OpenAIModel(
    model_name='llama3.1',
    provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)

# Define dependencies for the dataframe agent
@dataclass
class DataFrameDependencies:
    df: pd.DataFrame

class DataFrameQueryResult(BaseModel):
    answer: str = Field(description="The answer to the query about the dataframe")
    code_executed: str = Field(description="The pandas code that was executed to answer the query")

# Initialize the dataframe agent
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

# Function to process dataframe queries asynchronously
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
        return result
    except asyncio.TimeoutError:
        raise Exception("Processing took too long. Try a simpler question or smaller dataset.")

# Function to process CSV file
def process_csv(file):
    try:
        if file is None:
            return None, "Please upload a CSV file."
        # Ensure file pointer is at the beginning
        if hasattr(file, 'seek'):
            file.seek(0)
        # Read CSV file using the file path
        df = pd.read_csv(file.name)
        # Basic validation
        if df.empty:
            return None, "The uploaded CSV file is empty."
        return df, f"CSV loaded successfully. Shape: {df.shape}"
    except Exception as e:
        return None, f"Error processing CSV file: {str(e)}"

# Function to generate graph based on parameters
def generate_graph(df, graph_type, x_column, y_column=None, title="Graph", hue=None):
    if df is None:
        # Create a blank image with error text
        blank_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        pil_img = Image.fromarray(blank_img)
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
        draw.text((10, 10), "Please upload a CSV file first.", fill=(0, 0, 0), font=font)
        return np.array(pil_img)

    plt.figure(figsize=(10, 6))
    try:
        if graph_type.lower() == "bar":
            if y_column and y_column != "None":
                if hue and hue != "None":
                    sns.barplot(x=df[x_column], y=df[y_column], hue=df[hue])
                else:
                    sns.barplot(x=df[x_column], y=df[y_column])
            else:
                df[x_column].value_counts().plot(kind='bar')
        elif graph_type.lower() == "line":
            if y_column and y_column != "None":
                if hue and hue != "None":
                    sns.lineplot(x=df[x_column], y=df[y_column], hue=df[hue])
                else:
                    sns.lineplot(x=df[x_column], y=df[y_column])
            else:
                plt.plot(df[x_column])
        elif graph_type.lower() == "scatter":
            if y_column and y_column != "None":
                if hue and hue != "None":
                    sns.scatterplot(x=df[x_column], y=df[y_column], hue=df[hue])
                else:
                    sns.scatterplot(x=df[x_column], y=df[y_column])
            else:
                plt.scatter(range(len(df)), df[x_column])
        elif graph_type.lower() == "histogram":
            sns.histplot(df[x_column], kde=True)
        elif graph_type.lower() == "boxplot":
            if y_column and y_column != "None":
                if hue and hue != "None":
                    sns.boxplot(x=df[x_column], y=df[y_column], hue=df[hue])
                else:
                    sns.boxplot(x=df[x_column], y=df[y_column])
            else:
                sns.boxplot(y=df[x_column])
        elif graph_type.lower() == "heatmap":
            # For heatmap, we need numerical data
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_columns) < 2:
                raise ValueError("Not enough numeric columns for heatmap")
            corr_df = df[numeric_columns].corr()
            sns.heatmap(corr_df, annot=True, cmap="coolwarm")

        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close()

        # Convert buffer to image
        img = Image.open(buf)
        return np.array(img)
    except Exception as e:
        plt.close()
        # Create an error image
        error_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        error_pil = Image.fromarray(error_img)
        draw = ImageDraw.Draw(error_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
        # Format error message to fit on image
        error_msg = f"Error generating graph: {str(e)}"
        lines = []
        line = ""
        for word in error_msg.split():
            if len(line + word) + 1 <= 60:
                line = line + " " + word if line else word
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        # Draw error message
        y_position = 10
        for line in lines:
            draw.text((10, y_position), line, fill=(255, 0, 0), font=font)
            y_position += 20
        return np.array(error_pil)

# Function to answer questions about the data
async def answer_question(df, question):
    if df is None:
        return "Please upload a CSV file first.", None
    
    try:
        result = await process_dataframe_query(df, question)
        return result.answer, None
    except Exception as e:
        traceback_str = traceback.format_exc()
        return f"Error processing question: {str(e)}\n\n{traceback_str}", None

# Create Gradio interface
with gr.Blocks(title="CSV Question Answering and Visualization") as app:
    gr.Markdown("# CSV Question Answering and Visualization")
    
    # Global variable to store the dataframe
    df_state = gr.State(None)
    
    # File upload section
    with gr.Row():
        file_input = gr.File(label="Upload CSV File (Max 25MB)")
        load_button = gr.Button("Load CSV")
        status_output = gr.Textbox(label="Status", interactive=False)
    
    # Create tabs
    with gr.Tabs():
        # Tab 1: CSV Data Display
        with gr.TabItem("CSV Data"):
            data_preview = gr.Dataframe(label="Data Preview", interactive=False)
            column_info = gr.Dataframe(label="Column Information", interactive=False)
            stats_output = gr.Dataframe(label="Basic Statistics", interactive=False)
        
        # Tab 2: Question Answering
        with gr.TabItem("Ask Questions"):
            question_input = gr.Textbox(label="Ask a question about the data", placeholder="E.g., What is the average price?")
            question_button = gr.Button("Submit Question")
            answer_output = gr.Textbox(label="Answer", interactive=False)
        
        # Tab 3: Graph Generation
        with gr.TabItem("Create Graphs"):
            with gr.Row():
                with gr.Column():
                    graph_type = gr.Dropdown(
                        choices=["bar", "line", "scatter", "histogram", "boxplot", "heatmap"],
                        label="Graph Type"
                    )
                    x_column = gr.Dropdown(label="X-Axis Column")
                    y_column = gr.Dropdown(label="Y-Axis Column (optional for histogram)")
                    hue_column = gr.Dropdown(label="Hue/Group Column (optional)")
                    graph_title = gr.Textbox(label="Graph Title", value="Graph")
                    create_graph_button = gr.Button("Generate Graph")
                with gr.Column():
                    graph_output = gr.Image(label="Generated Graph")
    
    # Load CSV button click event
    def load_csv(file):
        df, message = process_csv(file)
        if df is not None:
            # Prepare column information
            column_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null Count': df.count().values,
                'Null Count': df.isna().sum().values
            })
            # Prepare statistics for numeric columns
            stats = df.describe().transpose().reset_index().rename(columns={'index': 'Column'})
            # Return new Dropdown objects with updated choices
            return df, message, df.head(10), column_df, stats, \
                gr.Dropdown(choices=df.columns.tolist()), \
                gr.Dropdown(choices=df.columns.tolist()), \
                gr.Dropdown(choices=['None'] + df.columns.tolist())
        else:
            return None, message, None, None, None, \
                gr.Dropdown(choices=[]), \
                gr.Dropdown(choices=[]), \
                gr.Dropdown(choices=[])
    
    load_button.click(
        fn=load_csv,
        inputs=[file_input],
        outputs=[df_state, status_output, data_preview, column_info, stats_output, x_column, y_column, hue_column]
    )
    
    # Submit question button click event
    async def process_question(df, question):
        answer, _ = await answer_question(df, question)
        return answer
    
    question_button.click(
        fn=process_question,
        inputs=[df_state, question_input],
        outputs=[answer_output]
    )
    
    # Create graph button click event
    def create_graph(df, graph_type, x_col, y_col, hue, title):
        if df is None:
            return "Please upload a CSV file first."
        # Handle 'None' selection for hue
        actual_hue = None if hue == 'None' else hue
        # Generate the graph
        return generate_graph(df, graph_type, x_col, y_col, title, actual_hue)
    
    create_graph_button.click(
        fn=create_graph,
        inputs=[df_state, graph_type, x_column, y_column, hue_column, graph_title],
        outputs=[graph_output]
    )

# Launch the app
if __name__ == "__main__":
    app.launch()
