#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.runners import InMemoryRunner
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, AgentTool, ToolContext, FunctionTool
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.models.google_llm import Gemini

import APIKey    #Storing my API key in python class "APIKey"
import os
google_API = APIKey.GOOGLESTUDIO_API_KEY    #My google studio API
os.environ["GOOGLE_API_KEY"] = google_API

import yfinance as yf
import pandas_datareader.data as web
import statsmodels.api as sm
import datetime as dt
import pandas_ta as ta
import asyncio

# --- 1. DEFINE TOOLS & FUNCTIONS ---

def competitors_compare(target: str) -> dict:
    """
    Identifies the target company's main competitors and returns a comparative financial table.

    Args:
        taregt: The stock ticker of a company (e.g., "AAPL", "NVDA"). 
                Must be comprised of uppercase letters.

    Returns:
        - On Success: {"status": "success", "comparison": markdown_table}
        - On Error: {"status": "error", "error_message": string}
    """

    target = target.upper()

    try:
        financial_ratio_df = pd.read_csv("financial_ratio.csv")
        financial_ratio_df = financial_ratio_df.sort_values(by='fiscalDateEnding', ascending=True)
        compare_df = financial_ratio_df.drop_duplicates(subset=['comp'], keep='last')

        compare_df = compare_df.replace("NaN", np.nan)
        compare_df = compare_df.dropna(axis=1, how='all')
        compare_df = compare_df.fillna("N/A")
        compare_df = compare_df.round(4)

        return {"status": "success", "comparison": compare_df.to_markdown(index=False)}

    except Exception as e:
        return {"status": "error", "error_message": str(e)}

def get_financial_ratio(target: str) -> dict:
    """
    Retrieves the target company's financial ratios for the last 5 years to analyze trends.

    Args:
        taregt: The stock ticker of a company (e.g., "AAPL", "NVDA"). 
                Must be comprised of uppercase letters.

    Returns:
        dict:
        - On Success: {"status": "success", "comparison": markdown_table}
        - On Error: {"status": "error", "error_message": string}
    """

    try:
        # Get data. Adjust it if doing general
        financial_ratio_df = pd.read_csv("financial_ratio.csv")

        target_ratio_df = financial_ratio_df[financial_ratio_df['comp'] == target].copy()
        if target_ratio_df.empty:
             return {
                 "status": "error", 
                 "error_message": f"Stock ticker '{target}' not found in the database."
             }
        target_ratio_df['fiscalDateEnding'] = pd.to_datetime(target_ratio_df['fiscalDateEnding'])
        target_ratio_df = target_ratio_df.sort_values(by='fiscalDateEnding', ascending=True)
        cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=5)
        target_ratio_df = target_ratio_df[target_ratio_df['fiscalDateEnding'] >= cutoff_date]

        target_ratio_df = target_ratio_df.replace("NA", np.nan)
        target_ratio_df = target_ratio_df.dropna(axis=1, how='all')
        target_ratio_df = target_ratio_df.fillna("N/A")
        target_ratio_df = target_ratio_df.round(4)

        return {
            "status": "success", 
            "financial_ratio": target_ratio_df.to_markdown(index=False)
        }

    except Exception as e:
        return {"status": "error", "error_message": str(e)}

def fair_value_calculation(target: str) -> str:
    """
    Calculates Fair Value using DCF.
    Returns a CLEAN STRING summary for the Aggregator Agent to read directly.
    """
    
    target = target.upper()
    ticker = yf.Ticker(target)

    lookback_years = 5
    default_growth_rate_projection = 0.00
    
    try:
        estimates = ticker.get_growth_estimates()
        if estimates is not None and not estimates.empty:
            growth_rate_projection = estimates.get('stockTrend', {}).get('+1y', default_growth_rate_projection)
        else:
            growth_rate_projection = default_growth_rate_projection
    except (KeyError, AttributeError, TypeError, ValueError) as e:
        growth_rate_projection = default_growth_rate_projection

    projection_years = 5
    terminal_growth_rate = 0.00
    
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=lookback_years*365)
    coe = 0.1

    try:
        ff_data = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start_date, end_date)[0]
        ff_data = ff_data / 100
        ff_data.index = ff_data.index.to_timestamp().to_period('M')
        
        stock_hist = ticker.history(start=start_date-pd.DateOffset(months=2), end=end_date, interval='1mo')
        stock_returns = stock_hist['Close'].pct_change().dropna().to_period('M')

        data = pd.merge(stock_returns, ff_data, left_index=True, right_index=True)
        data.columns = ['Stock_Return', 'Mkt-RF', 'SMB', 'HML', 'RF']
        data['Excess_Return'] = data['Stock_Return'] - data['RF']

        if len(data) > 24:
            X = sm.add_constant(data[['Mkt-RF', 'SMB', 'HML']])
            model = sm.OLS(data['Excess_Return'], X).fit()

            rf_mean = data['RF'].mean() * 12
            risk_premium = (model.params['Mkt-RF'] * data['Mkt-RF'].mean() + 
                            model.params['SMB'] * data['SMB'].mean() + 
                            model.params['HML'] * data['HML'].mean()) * 12
            coe = rf_mean + risk_premium
            
    except Exception:
        beta = ticker.info.get('beta', 1.0)
        coe = 0.03 + (beta * 0.05)

    # Get data. Adjust it if doing general
    try:
        financial_data_df = pd.read_csv("financial_data.csv")
        target_df = financial_data_df[financial_data_df['comp'] == target].copy()
        if target_df.empty:
            return f"Error: Ticker {target} not found in local financial database."
        target_df = target_df.sort_values(by='fiscalDateEnding', ascending=False)

        int_exp = target_df.iloc[0]['interestExpense']
        debt = target_df.iloc[0]['shortLongTermDebtTotal']
        tax_rate = target_df['taxRate'].mean()
        if debt > 0:
            cod = (int_exp / debt) * (1 - tax_rate)
        else:
            cod = 0.00

        shares = ticker.info.get('sharesOutstanding')
        current_price = ticker.info.get('currentPrice')

        if not shares or not current_price:
            return f"Error: Could not retrieve live price/shares for {target}."

        market_cap = shares * current_price
        total_value = market_cap + debt
        wacc = ((market_cap / total_value) * coe) + ((debt / total_value) * cod)
        if wacc <= terminal_growth_rate: 
            wacc = terminal_growth_rate + 0.01
    
    except Exception as e:
        return f"Error in valuation calculation: {str(e)}"

    net_income = target_df['netIncome']
    fcf = target_df['freeCashFlow']
    avg_fcf_conversion = (fcf / net_income).replace([np.inf, -np.inf], np.nan).mean()
    if np.isnan(avg_fcf_conversion) or avg_fcf_conversion < 0:
        return f"Error: Missing or negative FCF data for {target}."
        
    forward_eps = ticker.info.get('forwardEps') or ticker.info.get('trailingEps')
    start_fcf_per_share = forward_eps * avg_fcf_conversion

    future_fcf_values = []
    for i in range(1, projection_years + 1):
        val = start_fcf_per_share * ((1 + growth_rate_projection) ** i)
        future_fcf_values.append(val)
    terminal_value = (future_fcf_values[-1] * (1 + terminal_growth_rate)) / (wacc - terminal_growth_rate)
    discounted_fcfs = sum([val / ((1 + wacc) ** (i + 1)) for i, val in enumerate(future_fcf_values)])
    discounted_tv = terminal_value / ((1 + wacc) ** projection_years)
    intrinsic_value = discounted_fcfs + discounted_tv
    
    err_tolerence = 0.05
    upside = (intrinsic_value - current_price) / current_price
    if upside >err_tolerence: status = "UNDERVALUED"
    elif upside < -err_tolerence: status = "OVERVALUED"
    else: status = "FAIRLY VALUED"

    return f"""
    **DCF VALUATION MODEL RESULTS:**
    - **Status:** {status}
    - **Fair Value:** ${intrinsic_value:.2f} (Current: ${current_price:.2f})
    - **Upside/Downside:** {upside:.1%}
    - **Key Assumptions:** WACC {wacc:.1%}, {projection_years} Years Avg Growth {growth_rate_projection:.1%}, Terminal Growth {terminal_growth_rate:.1%}
    - **Methodology:** 5-Year DCF with Fama-French/CAPM Cost of Equity.
    """

def get_technical_analysis(target: str) -> dict:

    """
    Calculates key technical indicators for the target stock over the last 60 trading days.

    Args:
        target: The stock ticker of a company (e.g., "AAPL", "NVDA"). 
                Must be comprised of uppercase letters.

    Returns:
        dict:
        - On Success: {"status": "success", "comparison": markdown_table}
        - On Error: {"status": "error", "error_message": string}
    """
    try:
        lookback_years = 10
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=lookback_years*365)
        
        ticker = yf.Ticker(target)
        stock_hist = ticker.history(start=start_date, end=end_date, interval='1d', auto_adjust=True)
        if stock_hist.empty:
            return {"status": "error", "error_message": f"No price history found for {target}."}
        stock_hist.index = stock_hist.index.tz_localize(None)
        if stock_hist.index[-1].date() == dt.date.today():
            stock_hist = stock_hist.iloc[:-1]
        stock_hist = stock_hist.sort_index(ascending=True)
    
        technical_indicator = stock_hist[['Close', 'Volume']].copy()

        technical_indicator["RSI"] = ta.rsi(stock_hist["Close"], length=14)

        technical_indicator['MACD'] = ta.macd(stock_hist["Close"])['MACDh_12_26_9']

        adx = ta.adx(
            high=stock_hist["High"],
            low=stock_hist["Low"],
            close=stock_hist["Close"],
            length=14
        )
        technical_indicator['ADX'] = adx['ADX_14']
        technical_indicator['ADX_Dir'] = np.where(adx['DMP_14'] > adx['DMN_14'], "Bull", "Bear")

        technical_indicator["MFI"] = ta.mfi(
            high=stock_hist["High"],
            low=stock_hist["Low"],
            close=stock_hist["Close"],
            volume=stock_hist["Volume"],
            length=14
        )

        technical_indicator["Volume_Change"] = stock_hist["Volume"].pct_change() * 100

        technical_indicator['MA_10'] = ta.sma(stock_hist['Close'], length = 10)
        technical_indicator['MA_20'] = ta.sma(stock_hist['Close'], length = 20)
        technical_indicator['MA_50'] = ta.sma(stock_hist['Close'], length = 50)
        technical_indicator['MA_200'] = ta.sma(stock_hist['Close'], length = 200)

        sr = ta.pivots(open_=stock_hist['Open'], high=stock_hist['High'], low=stock_hist['Low'], close=stock_hist['Close'], method='traditional', anchor='D')
        technical_indicator['dist_to_S1'] = (stock_hist['Close'] - sr['PIVOTS_TRAD_D_S1']) / stock_hist['Close']
        technical_indicator['dist_to_R1'] = (stock_hist['Close'] - sr['PIVOTS_TRAD_D_R1']) / stock_hist['Close']

        analysis_days = 60
        technical_indicator = technical_indicator.tail(analysis_days)
        technical_indicator = technical_indicator.replace("nan", np.nan)
        technical_indicator = technical_indicator.dropna(axis=1, how='all')
        technical_indicator = technical_indicator.fillna("N/A")
        technical_indicator = technical_indicator.round(4)

        return {
            "status": "success", 
            "technical_indicator": technical_indicator.to_markdown(index=True)
        }
        
    except Exception as e:
        return {"status": "error", "error_message": str(e)}



# --- 2. DEFINE AGENT INITIALIZERS ---        

def news_agent_init():
    return Agent(
    name="GoogleNewsAgent",
    model="gemini-2.5-flash",
    instruction="""
    You are a Senior Market Intelligence Analyst. Your goal is to produce a concise, actionable news briefing for a specific stock.

    Step 1: FOUNDATION
    Use 'google_search' to find the company's latest '10-K business summary' or 'Investor Relations overview'. 
    - Goal: Understand strictly how they make money (e.g., "Revenue comes 60% from cloud, 40% from ads").

    Step 2: SENTIMENT SOURCING (CRITICAL: NEWEST DATA ONLY)
    Use 'google_search' to find news from the *last 14 days*.
    - Search Query Format: "[Company Ticker] stock news last 2 weeks" or "[Company Name] regulatory filings current month".
    - Specific focus: Financial reports, Management changes, Lawsuits, or Merge and acquisition.

    Step 3: ANALYSIS & MEMO
    Generate a formatted memo. Do not list links blindly. Group them into:
    - **Catalysts:** Concrete positive events.
    - **Risks:** Concrete negative threats.
    - **Market Sentiment:** Overall mood (Bullish/Bearish/Neutral).

    Constraint: If no relevant news exists in the last 14 days, explicitly state: "No significant material news in the last two weeks."
    """,
    tools = [google_search],
    output_key="google_news_arrangement",
)

def competitors_agent_init():
    return Agent(
    name="CompetitorsAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
    You are a Financial Peer Analysis Agent.

    1. RECEIVE DATA: 
       Call 'competitors_compare' with the user's target ticker.
       
    2. ERROR CHECK: 
       Check the "status" key. If it says "error", STOP and report the "error_message" to the user.

    3. ANALYZE (Only if status is "success"):
       The table contains the target and its specific competitors.
       - Compare the target (requested by user) against the others in the table.
       - Highlight where the target is stronger or weaker (Valuation, Margins, Debt).
    """,
    tools = [competitors_compare],
    output_key="comparing_competitors",
)

def financial_ratio_agent_init():
    return Agent(
    name="FinancialRatioAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
    You are a Fundamental Trend Analysis Agent.

    1. RECEIVE DATA: 
       Call 'get_financial_ratio' with the user's target ticker.
       
    2. ERROR CHECK: 
       Check the "status" key. If it says "error", STOP and report the "error_message" to the user.

    3. ANALYZE (Only if status is "success"):
       The dataset contains a wide range of financial ratios. 
       **You must analyze ALL metrics provided in the table.**
       Do not cherry-pick specific ratios unless they show significant anomalies.

       **Analysis Strategy:**
       - **Scan the entire table:** Look for *any* metric that shows a significant trend (consistently rising/falling) or a sudden spike/drop.
       - **Connect the dots:** Look for relationships between different ratios (e.g., "Inventory Turnover is slowing down while Current Ratio is rising—is the company hoarding unsold stock?").
       - **Categorize your findings:** Structure your analysis into these broad pillars using whichever metrics are available:
         * *Operational Efficiency* (How well do they use assets?)
         * *Profitability & Margins* (Are they making money?)
         * *Financial Solvency & Health* (Can they pay debts?)
         * *Growth & Valuation* (Is the growth real or expensive?)
    
    4. REPORT GENERATION:
       - Provide a detailed, structured analysis (not just a summary). 
       - Explicitly mention the *magnitude* of changes (e.g., "drastic drop" vs "slight fluctuation").
       - Highlight any divergences (e.g., "Revenue is up, but Cash Flow is down").
    """,
    tools = [get_financial_ratio],
    output_key="financial_ratio_analysis",
)

def technical_agent_init():
    return Agent(
    name="TechnicalAnalysisAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
    You are a Technical Analysis Expert (CMT).

    1. DATA RETRIEVAL:
       Call 'get_technical_analysis' to retrieve the recent price action and indicator history.

    2. ANALYSIS STRATEGY:
       Review the dataset to determine the stock's current technical posture. 
       **You have full autonomy to determine which indicators are most relevant for the current market structure.**

       - Look for confluence between Trend, Momentum, and Volatility.
       - Identify key structural shifts (e.g., breakouts, breakdowns, divergences, or consolidations).
       - Assess the "Character" of the recent price action (e.g., is the buying volume convincing? Is the trend exhausting?).

    3. OUTPUT:
       Synthesize your findings into a professional technical memo. 
       - Avoid defining standard terms (assume the reader knows what RSI is).
       - Focus purely on the **implications** of the data.
       - Conclude with a clear directional bias: Bullish, Bearish, or Neutral/Awaiting setup.
    """,
    tools = [get_technical_analysis],
    output_key="technical_analysis",
)

# --- 3. MAIN EXECUTION FLOW ---

def simple_run(runner, prompt_text):
    
    user_id = "user_01"

    session = asyncio.run(runner.session_service.create_session(
        app_name='Financial_Analysis', user_id=user_id
    ))
    
    message = types.Content(role="user", parts=[types.Part(text=prompt_text)])
    
    # Sync Generator call
    response_generator = runner.run(
        user_id=user_id,
        session_id=session.id,
        new_message=message
    )
    
    final_text = []
    try:
        for event in response_generator:
            if event.content.parts and event.content.parts[0].text:
                final_text.append(event.content.parts[0].text)
    except Exception as e:
        return f"Error reading generator: {e}"
            
    return "".join(final_text)

def run_stock_analysis(target):
    
    news = news_agent_init()
    comp = competitors_agent_init()
    fin_r = financial_ratio_agent_init()
    tech = technical_agent_init()
    
    dcf_result = fair_value_calculation(target)
    
    aggregator =  Agent(
    name="AggregatorAgent",
    model="gemini-2.5-flash",
    instruction=f"""
    You are a Portfolio Manager writing a Final Investment Memo.

    INPUT DATA:
    **News:** {{google_news_arrangement}}
    **Comparison:** {{comparing_competitors}}
    **Financial Ratio:** {{financial_ratio_analysis}}
    **Valuation Model (DCF):** {dcf_result}
    **Technical Indicators:** {{technical_analysis}}
    
    TASK:
    Synthesize these 5 inputs into a cohesive, 300-word strategic note.
    **Crucial: If the input contradict each other, explicitly address this conflict in the "Risks" or "Verdict" section.**

    STRUCTURE:
    - **Executive Summary:** The "Bottom Line" (Buy/Sell/Hold) and the primary driver.
    - **Advantages & Disadvantages:** (Balance the growth potential against the financial health).
    - **Risks:** (Focus on downside scenarios and data conflicts).
    - **Final Verdict:** Clear recommendation with a target price reference.

    TONE:
    Professional, objective, and decisive.
    """,
    output_key="investment_memo",
    )
  
    parallel_analysis_team = ParallelAgent(
    name="ParallelAnalysisTeam",
    sub_agents=[news, comp, fin_r, tech],
    )

    root_agent = SequentialAgent(
        name="AnalysisSystem",
        sub_agents=[parallel_analysis_team, aggregator],
    )

    runner = InMemoryRunner(agent=root_agent, app_name='Financial_Analysis')
    final_memo = simple_run(runner, f"Analyze {target}")
            
    return final_memo


# In[ ]:




