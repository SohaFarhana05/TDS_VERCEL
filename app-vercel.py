import os
import sys
import json
import base64
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re
import requests
from io import BytesIO
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from dotenv import load_dotenv
import networkx as nx

# Simplified LangChain imports
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.tools import tool
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

load_dotenv()

app = FastAPI(title="TDS Data Analyst Agent - Vercel")

# Simplified LLM configuration
GEMINI_KEYS = [os.getenv(f"gemini_api_{i}") for i in range(1, 11)]
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]

if not GEMINI_KEYS:
    print("Warning: No Gemini API keys found")

# Simplified LLM class
class SimpleLLM:
    def __init__(self):
        self.api_key = GEMINI_KEYS[0] if GEMINI_KEYS else None
        
    def get_response(self, prompt: str) -> str:
        if not self.api_key:
            return "Error: No API key configured"
        
        try:
            # Simple direct API call to Gemini
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

llm = SimpleLLM()

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML interface"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found</h1><p>Please ensure index.html is in the same directory as app.py</p>", status_code=404)

@app.post("/analyze")
async def analyze_data(
    questions_file: UploadFile = File(...),
    data_file: UploadFile = File(None)
):
    """Simplified analysis endpoint"""
    try:
        # Read questions file
        questions_content = await questions_file.read()
        questions = questions_content.decode('utf-8')
        
        # Read data file if provided
        data_df = None
        if data_file:
            data_content = await data_file.read()
            if data_file.filename.endswith('.csv'):
                data_df = pd.read_csv(BytesIO(data_content))
            elif data_file.filename.endswith('.json'):
                data_df = pd.read_json(BytesIO(data_content))
        
        # Simple analysis prompt
        analysis_prompt = f"""
        Analyze the following questions and data:
        
        Questions: {questions[:1000]}  # Limit to avoid token limits
        
        Data summary: {data_df.describe().to_string() if data_df is not None else 'No data provided'}
        
        Provide insights and analysis.
        """
        
        # Get LLM response
        response = llm.get_response(analysis_prompt)
        
        # Generate simple visualization if data is provided
        charts = []
        if data_df is not None:
            try:
                # Create a simple plot
                plt.figure(figsize=(10, 6))
                if len(data_df.columns) > 1:
                    numeric_cols = data_df.select_dtypes(include=[np.number]).columns[:2]
                    if len(numeric_cols) >= 2:
                        plt.scatter(data_df[numeric_cols[0]], data_df[numeric_cols[1]])
                        plt.xlabel(numeric_cols[0])
                        plt.ylabel(numeric_cols[1])
                        plt.title(f'{numeric_cols[0]} vs {numeric_cols[1]}')
                    else:
                        data_df.iloc[:, 0].hist()
                        plt.title('Data Distribution')
                else:
                    data_df.iloc[:, 0].hist()
                    plt.title('Data Distribution')
                
                # Convert plot to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                buffer.seek(0)
                chart_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                charts.append({
                    "title": "Data Visualization",
                    "chart": f"data:image/png;base64,{chart_base64}"
                })
            except Exception as e:
                print(f"Chart generation error: {e}")
        
        return {
            "status": "success",
            "analysis": response,
            "charts": charts,
            "data_summary": data_df.describe().to_dict() if data_df is not None else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "TDS Data Analyst Agent"}

@app.get("/summary")
async def get_summary():
    """System summary endpoint"""
    return {
        "status": "ok",
        "service": "TDS Data Analyst Agent - Vercel Edition",
        "features": ["Data Analysis", "Visualization", "LLM Integration"],
        "api_keys_configured": len(GEMINI_KEYS)
    }

# Vercel handler
handler = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
