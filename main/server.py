import pandas as pd
import yfinance as yf
import datetime as dt
import os
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import agent

app = FastAPI()

report_storage = {}
backtest_storage = {}

class TickerRequest(BaseModel):
    ticker: str

# --- 1. Active Server ---
@app.on_event("startup")
def startup_event():
    print("Server starting... ensuring financial ratios are calculated.")

# --- 2. Basic Terminal ---
@app.get("/")
def home():
    return {"message": "Stock Analysis API is running"}

@app.get("/api/search")
def search(keyword: str):
    keyword = keyword.upper()
    return {"data": [
        {"symbol": keyword, "name": f"{keyword} (User Input)"},
    ]}

# --- 3. Investment Memo ---
@app.post("/api/analyze_ai/{ticker}")
def analyze_ai_endpoint(ticker: str, date: str = None, terminal_growth_rate: float = 0.02):
    try:
        print(f"Starting AI analysis workflow for {ticker}...")

        if ticker == "SHEL":
            fundamental_data = agent.temp_fin_data()
            financial_ratio = agent.temp_fin_ratio()
        else:
            fundamental_data = agent.process_fundamental_data(ticker)
            financial_ratio = agent.calculate_financial_ratio(fundamental_data)
        
        price_data = agent.process_technical_data(ticker, date)
        technical_indicator = agent.calculate_technical_indicator(price_data)

        memo = agent.run_stock_analysis(date, ticker, fundamental_data, financial_ratio, technical_indicator, terminal_growth_rate)
        
        report_storage[ticker] = {
            "date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "financial_ratio": financial_ratio,
            "price_data": price_data,
            "technical_indicator": technical_indicator,
            "analysis": memo  
        }
        
        print(f"Analysis for {ticker} completed and stored.")
        return {"status": "success", "message": "Analysis generated"}
        
    except Exception as e:
        print(f"Error during AI analysis: {e}")
        raise HTTPException(status_code=500, detail=f"AI Processing Error: {str(e)}")

@app.get("/api/get_ai_report/{ticker}")
def get_ai_report(ticker: str):
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

# --- 4. Fundamental Analysis ---
@app.get("/api/fundamental/{ticker}")
def get_fundamental_data(ticker: str):
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


# --- 5. Technical Analysis ---
@app.get("/api/technical/{ticker}")
def analyze_technical_endpoint(ticker: str):
    report = report_storage.get(ticker)
    if not report:
        raise HTTPException(status_code=404, detail="Ticker not found.")
    
    price_data = report.get("price_data")
    technical_indicator = report.get("technical_indicator")
    if technical_indicator.empty or price_data.empty:
        raise HTTPException(status_code=404, detail="Data fetching pending or failed.")
    
    price_data = price_data.reset_index()
    if 'Date' not in price_data.columns and 'index' in price_data.columns:
        price_data.rename(columns={'index': 'Date'}, inplace=True)
    price_data = json.loads(price_data.to_json(orient='records', date_format='iso'))

    technical_indicator = technical_indicator.reset_index()
    if 'Date' not in technical_indicator.columns and 'index' in technical_indicator.columns:
        technical_indicator.rename(columns={'index': 'Date'}, inplace=True)
    technical_indicator = json.loads(technical_indicator.to_json(orient='records', date_format='iso'))

    return {
        "date": report.get("date"),
        "price_data": price_data,
        "technical_indicator": technical_indicator
    } 

# --- 6. Backtesting ---
@app.get("/api/backtest/{ticker}")
def backtesting(ticker: str, date: str = None, terminal_growth_rate: float = 0.02):
    try:
        if date is None:
            valuation_date = dt.date.today() - pd.DateOffset(years=1)
            date_str = valuation_date.strftime("%Y-%m-%d")
        else:
            date_str = date
            valuation_date = dt.datetime.strptime(date_str, "%Y-%m-%d")

        print(f"Starting AI analysis workflow for {ticker} on {date_str}...")

        if ticker == "SHEL":
            fundamental_data = agent.temp_fin_data()
            financial_ratio = agent.temp_fin_ratio()
        else:
            fundamental_data = agent.process_fundamental_data(ticker)
            financial_ratio = agent.calculate_financial_ratio(fundamental_data)
        
        price_data = agent.process_technical_data(ticker, date_str)
        technical_indicator = agent.calculate_technical_indicator(price_data)

        memo = agent.run_stock_analysis(
            date_str, 
            ticker, 
            fundamental_data, 
            financial_ratio, 
            technical_indicator, 
            terminal_growth_rate
        )
        
        storage_key = f"{ticker}_{date_str}"
        backtest_storage[storage_key] = {
            "date": date_str,
            "analysis": memo  
        }
        
        print(f"Backtest stored with key: {storage_key}.")
        return {"status": "success", "message": "Analysis generated"}
        
    except Exception as e:
        print(f"Error during AI analysis: {e}")
        raise HTTPException(status_code=500, detail=f"AI Processing Error: {str(e)}")

@app.get("/api/get_backtest_report/{ticker}")
def get_backtest_report(ticker: str, date: str):
    storage_key = f"{ticker}_{date}"
    report = backtest_storage.get(storage_key)
    if not report:
        print(f"Failed to find key: {storage_key}")
        raise HTTPException(status_code=404, detail="Ticker not found.")
    return {
        "date": report.get("date"),
        "analysis": report.get("analysis")
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
