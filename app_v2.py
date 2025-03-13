import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import List, Dict, Any, Optional, Union
import numpy as np
import traceback
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Initialize the LLM agent with Ollama
ollama_model = OpenAIModel(
    model_name='llama3.1', 
    provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)

agent = Agent(model=ollama_model)

# Define Pydantic models for structured responses
class GraphRequest(BaseModel):
    """Request to generate a graph from CSV data"""
    graph_type: str = Field(description="Type of graph (bar, line, scatter, histogram, etc.)")
    x_column: str = Field(description="Column to use for x-axis")
    y_column: Optional[str] = Field(None, description="Column to use for y-axis (optional for some graph types)")
    title: str = Field(description="Title of the graph")
    color: Optional[str] = Field(None, description="Color to use for the graph")
    hue: Optional[str] = Field(None, description="Column to use for grouping (for seaborn plots)")

class DataQuery(BaseModel):
    """Query about data in the CSV file"""
    query_type: str = Field(description="Type of query (statistical, filtering, etc.)")
    columns: List[str] = Field(description="Columns involved in the query")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filters to apply")
    aggregation: Optional[str] = Field(None, description="Aggregation function (mean, sum, etc.)")

class QueryResponse(BaseModel):
    """Response to a query about CSV data"""
    answer: str = Field(description="Textual answer to the query")
    graph_request: Optional[GraphRequest] = Field(None, description="Graph request if visualization is appropriate")

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

# Function to generate graph based on GraphRequest
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
        return "Please upload a CSV file first."
    
    try:
        # Get dataframe info for context
        buffer = io.StringIO()
        df.info(buf=buffer)
        df_info = buffer.getvalue()
        
        # Get basic statistics
        df_stats = df.describe().to_string()
        
        # Get column names and sample data
        columns = df.columns.tolist()
        sample_data = df.head(5).to_string()
        
        # Prepare context for the LLM
        context = f"""
        DataFrame Information:
        {df_info}
        
        Basic Statistics:
        {df_stats}
        
        Columns: {columns}
        
        Sample Data:
        {sample_data}
        """
        
        # Use Pydantic AI to get structured response
        prompt = f"Based on the CSV data provided, answer this question: {question}. If appropriate, suggest a visualization.\n\nContext:\n{context}"
        response = await agent.run(prompt, result_type=QueryResponse)
        
        return response.answer, response.graph_request
    
    except Exception as e:
        traceback_str = traceback.format_exc()
        return f"Error processing question: {str(e)}\n\n{traceback_str}", None

# Create Gradio interface
with gr.Blocks(title="CSV Question Answering and Visualization") as app:
    gr.Markdown("# CSV Question Answering and Visualization")
    
    # Global variable to store the dataframe
    df_state = gr.State(None)
    
    # File upload section (common to all tabs)
    with gr.Row():
        file_input = gr.File(label="Upload CSV File (Max 25MB)")
        load_button = gr.Button("Load CSV")
        status_output = gr.Textbox(label="Status", interactive=False)
    
    # Create tabs
    with gr.Tabs():
        # Tab 1: CSV Data Display
        with gr.TabItem("CSV Data"):
            data_preview = gr.Dataframe(label="Data Preview", interactive=False)
            
            # Add column information display
            column_info = gr.Dataframe(label="Column Information", interactive=False)
            
            # Add basic statistics
            stats_output = gr.Dataframe(label="Basic Statistics", interactive=False)
        
        # Tab 2: Question Answering
        with gr.TabItem("Ask Questions"):
            question_input = gr.Textbox(label="Ask a question about the data", placeholder="E.g., What is the average price?")
            question_button = gr.Button("Submit Question")
            answer_output = gr.Textbox(label="Answer", interactive=False)
            
            # Add a section to display suggested visualization if available
            suggested_viz = gr.Image(label="Suggested Visualization", visible=False)
            viz_info = gr.Textbox(label="Visualization Details", visible=False)
        
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
            
            # Return new Dropdown objects with updated choices instead of using .update
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
        answer, graph_request = await answer_question(df, question)
        
        if graph_request:
            # Generate the suggested visualization
            graph_img = generate_graph(
                df, 
                graph_request.graph_type, 
                graph_request.x_column, 
                graph_request.y_column, 
                graph_request.title, 
                graph_request.hue
            )
            
            viz_details = f"Suggested visualization: {graph_request.graph_type} chart with x={graph_request.x_column}, y={graph_request.y_column if graph_request.y_column else 'N/A'}, hue={graph_request.hue if graph_request.hue else 'N/A'}"
            
            return answer, graph_img, True, viz_details, True
        else:
            return answer, None, False, "", False
    
    question_button.click(
        fn=process_question,
        inputs=[df_state, question_input],
        outputs=[answer_output, suggested_viz, suggested_viz, viz_info, viz_info]
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
