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

load_dotenv()

app = FastAPI(title="TDS Data Analyst Agent - Vercel")

# Simple LLM configuration
GEMINI_KEYS = [os.getenv(f"gemini_api_{i}") for i in range(1, 11)]
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]

if not GEMINI_KEYS:
    print("Warning: No Gemini API keys found")

# Simple LLM class
class SimpleLLM:
    def __init__(self):
        self.api_key = GEMINI_KEYS[0] if GEMINI_KEYS else None
        
    def get_response(self, prompt: str) -> str:
        if not self.api_key:
            return "Error: No API key configured"
        
        try:
            # Simple HTTP request to Gemini API
            import requests
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.api_key}"
            headers = {'Content-Type': 'application/json'}
            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    return result['candidates'][0]['content']['parts'][0]['text']
            
            return f"API Error: {response.status_code} - {response.text[:100]}"
        except Exception as e:
            return f"Error: {str(e)}"

llm = SimpleLLM()

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML interface"""
    try:
        # Simple HTML response for Vercel
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>TDS Data Analyst Agent</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #333; text-align: center; }
                .upload-section { margin: 20px 0; padding: 20px; border: 2px dashed #ccc; border-radius: 5px; }
                input[type="file"] { margin: 10px 0; }
                button { background: #007cba; color: white; border: none; padding: 12px 20px; border-radius: 5px; cursor: pointer; }
                button:hover { background: #005a8b; }
                .result { margin-top: 20px; padding: 20px; background: #f9f9f9; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 TDS Data Analyst Agent</h1>
                <p>Upload your questions file and optional dataset to get intelligent analysis with visualizations</p>
                
                <form id="analysisForm" enctype="multipart/form-data">
                    <div class="upload-section">
                        <label><strong>Questions File (.txt) *</strong></label><br>
                        <input type="file" id="questionsFile" accept=".txt" required>
                    </div>
                    
                    <div class="upload-section">
                        <label><strong>Dataset (Optional)</strong></label><br>
                        <input type="file" id="dataFile" accept=".csv,.json">
                    </div>
                    
                    <button type="submit">🔍 Analyze Data</button>
                </form>
                
                <div id="result" class="result" style="display:none;">
                    <h3>Analysis Results:</h3>
                    <div id="analysisContent"></div>
                </div>
            </div>

            <script>
                document.getElementById('analysisForm').addEventListener('submit', async function(e) {
                    e.preventDefault();
                    
                    const formData = new FormData();
                    const questionsFile = document.getElementById('questionsFile').files[0];
                    const dataFile = document.getElementById('dataFile').files[0];
                    
                    if (!questionsFile) {
                        alert('Please select a questions file');
                        return;
                    }
                    
                    formData.append('questions_file', questionsFile);
                    if (dataFile) {
                        formData.append('data_file', dataFile);
                    }
                    
                    try {
                        const response = await fetch('/analyze', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const result = await response.json();
                        
                        if (result.status === 'success') {
                            document.getElementById('analysisContent').innerHTML = 
                                '<h4>Analysis:</h4><pre>' + result.analysis + '</pre>' +
                                (result.charts && result.charts.length > 0 ? 
                                    '<h4>Visualizations:</h4>' + 
                                    result.charts.map(chart => '<img src="' + chart.chart + '" style="max-width:100%;margin:10px 0;">').join('') 
                                    : '');
                            document.getElementById('result').style.display = 'block';
                        } else {
                            throw new Error(result.detail || 'Analysis failed');
                        }
                    } catch (error) {
                        alert('Error: ' + error.message);
                    }
                });
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error loading frontend</h1><p>{str(e)}</p>", status_code=500)

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
        if data_file and data_file.filename:
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
        
        Provide insights and analysis in a clear, structured format.
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

# Vercel handler
handler = app
