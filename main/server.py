import pandas as pd
import yfinance as yf
import datetime
import os
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import agent

app = FastAPI()

report_storage = {}

class TickerRequest(BaseModel):
    ticker: str

# --- 1. 伺服器啟動事件 ---
@app.on_event("startup")
def startup_event():
    """
    伺服器啟動時，自動執行資料處理。
    這會讀取 financial_data.csv 並生成 financial_ratio.csv。
    """
    print("🔄 Server starting... ensuring financial ratios are calculated.")
    # agent.process_financial_data()

# --- 2. 基礎端點 ---
@app.get("/")
def home():
    return {"message": "Stock Analysis API is running"}

@app.get("/api/search")
def search(keyword: str):
    """模擬搜尋功能，回傳符合的股票代碼 (對應前端搜尋框)"""
    keyword = keyword.upper()
    # 這裡可以實作更複雜的搜尋，目前回傳包含關鍵字的範例
    return {"data": [
        {"symbol": keyword, "name": f"{keyword} (User Input)"},
    ]}

# --- 3. AI 分析端點 (Investment Memo) ---
@app.post("/api/analyze_ai/{ticker}")
def analyze_ai_endpoint(ticker: str):
    """
    觸發 AI 完整分析流程。
    這會呼叫 agent.py 中的 run_analysis_workflow。
    """
    try:
        print(f"🚀 Starting AI analysis workflow for {ticker}...")
        
        # 呼叫 Agent 進行分析 (這可能需要一點時間)
        if ticker == "SHEL":
            fundamental_data = agent.temp_fin_data()
            financial_ratio = agent.temp_fin_ratio()
        else:
            fundamental_data = agent.process_fundamental_data(ticker)
            financial_ratio = agent.calculate_financial_ratio(fundamental_data)
        
        price_data = agent.process_technical_data(ticker)
        technical_indicator = agent.calculate_technical_indicator(price_data)

        memo = agent.run_stock_analysis(ticker, fundamental_data, financial_ratio, technical_indicator)
        
        # 將結果存入暫存區，標記時間
        report_storage[ticker] = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "financial_ratio": financial_ratio,
            "technical_indicator": technical_indicator,
            "analysis": memo  # app.py 前端預設讀取 'news_analysis' 欄位來顯示結果
        }
        
        print(f"✅ Analysis for {ticker} completed and stored.")
        return {"status": "success", "message": "Analysis generated"}
        
    except Exception as e:
        print(f"❌ Error during AI analysis: {e}")
        # 如果失敗，回傳 500 錯誤給前端
        raise HTTPException(status_code=500, detail=f"AI Processing Error: {str(e)}")

@app.get("/api/get_ai_report/{ticker}")
def get_ai_report(ticker: str):
    """取得已生成的 AI 報告"""
    report = report_storage.get(ticker)
    if not report:
        raise HTTPException(status_code=404, detail="Ticker not found.")
    memo = report.get("analysis")
    if not memo:
        raise HTTPException(status_code=404, detail="Analysis pending or failed.")
    return {
        "date": report.get("date"),
        "analysis": memo
    }

# --- 4. 基本面數據端點 (Financial Stats Tab) ---
@app.get("/api/fundamental/{ticker}")
def get_fundamental_data(ticker: str):
    """
    讀取 CSV 數據並回傳給前端畫圖。
    優先讀取計算過比率的 financial_ratio.csv。
    """
    report = report_storage.get(ticker)
    if not report:
        raise HTTPException(status_code=404, detail="Ticker not found.")
    financial_ratio = report.get("financial_ratio")
    if financial_ratio.empty:
        raise HTTPException(status_code=404, detail="Data fetching pending or failed.")
    financial_ratio = json.loads(financial_ratio.to_json(orient='records', date_format='iso'))
    return {
        "date": report.get("date"),
        "financial_ratio": financial_ratio
    } 


# --- 6. 技術分析端點 (Technical Tab) ---
@app.get("/api/technical/{ticker}")
def analyze_technical_endpoint(ticker: str):
    """
    對應前端 Tab 5 的 'Run Technical Analysis' 按鈕。
    直接呼叫 agent.py 中的工具函數。
    """
    report = report_storage.get(ticker)
    if not report:
        raise HTTPException(status_code=404, detail="Ticker not found.")
    technical_indicator = report.get("technical_indicator")
    if technical_indicator.empty:
        raise HTTPException(status_code=404, detail="Data fetching pending or failed.")
    technical_indicator = technical_indicator.reset_index()
    if 'Date' not in technical_indicator.columns and 'index' in technical_indicator.columns:
        technical_indicator.rename(columns={'index': 'Date'}, inplace=True)
    technical_indicator = json.loads(technical_indicator.to_json(orient='records', date_format='iso'))
    return {
        "date": report.get("date"),
        "technical_indicator": technical_indicator
    } 

if __name__ == "__main__":
    # 啟動伺服器，監聽所有 IP，Port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
