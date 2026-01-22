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
google_API = APIKey.GOOGLESTUDIO_API_KEY   #My google studio API
alphavantage_API = APIKey.ALPHAVANTAGE_API_KEY
os.environ["GOOGLE_API_KEY"] = google_API

import yfinance as yf
import pandas_datareader.data as web
import statsmodels.api as sm
import datetime as dt
import pandas_ta as ta
import asyncio
import requests
import time
import random

# --- 1. MAIN DATA FETCHING ---
def temp_fin_data():
    return pd.read_csv("financial_data.csv")
def temp_fin_ratio():
    return pd.read_csv("financial_ratio.csv")

def process_fundamental_data(target: str):

    ticker = yf.Ticker(target)
    time.sleep(random.uniform(2, 5))

    try:
        industry_key = ticker.info.get('industryKey')
        if not industry_key:
            print(f"Warning: 'industryKey' not found for {target}. Using target only.")
            competitors = [target]
        else:
            industry = yf.Industry(industry_key)
            competitors_num = 3
            competitors = list(industry.top_companies.index.values)
            competitors = [c for c in competitors if c.upper() != target.upper()]
            competitors = competitors[:competitors_num]
            competitors.insert(0, target)
    except Exception as e:
        print(f"Error fetching competitors: {e}")
        competitors = [target]

    competitor_data_list = []

    for comp in competitors[:]:
        try:
            print(f"Fetching data for {comp}...")
        
            # 1. Fetch Financial Statement
            is_url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={comp}&apikey={alphavantage_API}'
            is_data = requests.get(is_url).json()
            # Wait 12 seconds to respect the 5 calls/minute limit (60s / 5 = 12s per call)
            time.sleep(12) 

            bs_url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={comp}&apikey={alphavantage_API}'
            bs_data = requests.get(bs_url).json()
            time.sleep(12)

            cf_url = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={comp}&apikey={alphavantage_API}'
            cf_data = requests.get(cf_url).json()
            time.sleep(12)

            if 'annualReports' not in is_data or 'annualReports' not in bs_data or 'annualReports' not in cf_data:
                print(f"Removing {comp}: Data incomplete or API limit reached.")
                competitors.remove(comp) 
                continue

            is_df = pd.json_normalize(is_data['annualReports'])
            bs_df = pd.json_normalize(bs_data['annualReports'])
            cf_df = pd.json_normalize(cf_data['annualReports'])

            merged_df = pd.merge(is_df, bs_df, on=['fiscalDateEnding', 'reportedCurrency'], how='outer', suffixes=('', '_bs'))
            merged_df = pd.merge(merged_df, cf_df, on=['fiscalDateEnding', 'reportedCurrency'], how='outer', suffixes=('', '_cf'))
            cols_to_drop = [c for c in merged_df.columns if c.endswith('_bs') or c.endswith('_cf')]
            merged_df.drop(columns=cols_to_drop, inplace=True)

            merged_df['fiscalDateEnding'] = pd.to_datetime(merged_df['fiscalDateEnding'])
            merged_df = merged_df.sort_values('fiscalDateEnding')

            # 2. Fetch Dividend Data
            div_url = f'https://www.alphavantage.co/query?function=DIVIDENDS&symbol={comp}&apikey={alphavantage_API}'
            div_data = requests.get(div_url).json()
            time.sleep(12)

            if 'data' in div_data and div_data['data']:
                div_df = pd.json_normalize(div_data['data'])
                div_df['ex_dividend_date'] = pd.to_datetime(div_df['ex_dividend_date'])
                div_df['amount'] = pd.to_numeric(div_df['amount'], errors='coerce')
            else:
                print(f"Note: No dividend data found for {comp} (or API limit). Assuming 0 dividends.")
                div_df = pd.DataFrame(columns=['ex_dividend_date', 'amount'])

            merged_df['start_date_calc'] = merged_df['fiscalDateEnding'].shift(1).fillna(pd.Timestamp("1900-01-01"))

            def calculate_accumulated_div(row):
                if div_df.empty:
                    return 0.0
                mask = (
                    (div_df['ex_dividend_date'] > row['start_date_calc']) & 
                    (div_df['ex_dividend_date'] <= row['fiscalDateEnding'])
                )
                matching_divs = div_df.loc[mask, 'amount']
                return matching_divs.sum()
            
            merged_df['dividend'] = merged_df.apply(calculate_accumulated_div, axis=1)
            merged_df.drop(columns=['start_date_calc'], inplace=True)

            # 3. Fetch Historical Stock Price Data
            if not merged_df.empty:
                min_date = merged_df['fiscalDateEnding'].min() - pd.Timedelta(days=7)
                hist = yf.download(comp, start=min_date, progress=False, auto_adjust=False)
            
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)
            
                hist = hist.reset_index()

                if 'Close' in hist.columns and 'Date' in hist.columns:
                    price_df = hist[['Date', 'Close']].copy()
                    price_df.rename(columns={'Date': 'price_date', 'Close': 'closePrice'}, inplace=True)
                
                    price_df['price_date'] = pd.to_datetime(price_df['price_date'])
                    price_df['closePrice'] = price_df['closePrice'].round(2)
                    price_df = price_df.sort_values('price_date')
                
                    merged_df = pd.merge_asof(
                        merged_df,
                        price_df,
                        left_on='fiscalDateEnding',
                        right_on='price_date',
                        direction='backward' # Backward = Look for price on that day, or previous available day
                    )
                else:
                    merged_df['closePrice'] = np.nan

            merged_df.insert(0, 'comp', comp)
            competitor_data_list.append(merged_df)
            print(f"Successfully processed {comp}")

        except Exception as e:
            print(f"Error processing {comp}: {e}")

    if competitor_data_list:

        competitor_df = pd.concat(competitor_data_list, ignore_index=True)

        cols_to_exclude = ['comp', 'fiscalDateEnding', 'reportedCurrency']
        cols_to_convert = [c for c in competitor_df.columns if c not in cols_to_exclude]
        
        competitor_df[cols_to_convert] = competitor_df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

        competitor_df['reportedEPS'] = np.where(competitor_df['commonStockSharesOutstanding'] != 0, 
                                                competitor_df['netIncome'] / competitor_df['commonStockSharesOutstanding'], np.nan)
        competitor_df['taxRate'] = np.where(competitor_df['incomeBeforeTax'] != 0, 
                                            competitor_df['incomeTaxExpense'] / competitor_df['incomeBeforeTax'], np.nan)
        competitor_df['freeCashFlow'] = (competitor_df['operatingCashflow'] + 
                                         competitor_df['interestExpense'] * (1 - competitor_df['taxRate']) + 
                                         competitor_df['cashflowFromInvestment'])
    else:
        print("No data found")
        competitor_df = pd.DataFrame()

    return competitor_df

def calculate_financial_ratio(financial_data: pd.DataFrame):

    financial_data = financial_data.sort_values(by=['comp', 'fiscalDateEnding'], ascending=[True, True])
    financial_ratio = financial_data[['comp', 'fiscalDateEnding']].copy()

    # 1. Profitability
    financial_ratio['GrossMargin'] = np.where(
        financial_data['totalRevenue'] != 0,
        financial_data['grossProfit'] / financial_data['totalRevenue'],
        np.nan
    )
    financial_ratio['OperatingMargin'] = np.where(
        financial_data['totalRevenue'] != 0,
        financial_data['operatingIncome'] / financial_data['totalRevenue'],
        np.nan
    )
    financial_ratio['NetProfitMargin'] = np.where(
        financial_data['totalRevenue'] != 0,
        financial_data['netIncome'] / financial_data['totalRevenue'],
        np.nan
    )
    financial_ratio['ROE'] = np.where(
        financial_data['totalShareholderEquity'] != 0,
        financial_data['netIncome'] / financial_data['totalShareholderEquity'],
        np.nan
    )
    financial_ratio['ROA'] = np.where(
        financial_data['totalAssets'] != 0,
        financial_data['netIncome'] / financial_data['totalAssets'],
        np.nan
    )
    ## Cancel ROIC

    # 2. Leverage
    financial_ratio['D/E'] = np.where(
        financial_data['totalShareholderEquity'] != 0,
        financial_data['totalLiabilities'] / financial_data['totalShareholderEquity'],
        np.nan
    )
    financial_ratio['CurrentRatio'] = np.where(
        financial_data['totalCurrentLiabilities'] != 0,
        financial_data['totalCurrentAssets'] / financial_data['totalCurrentLiabilities'],
        np.nan
    )
    financial_ratio['InterestCoverage'] = np.where(
        financial_data['interestExpense'] != 0,
        financial_data['ebit'] / financial_data['interestExpense'],
        np.nan
    )
    financial_ratio['Debt/EBITDA'] = np.where(
        financial_data['ebitda'] != 0,
        financial_data['totalLiabilities'] / financial_data['ebitda'],
        np.nan
    )

    # 3. Efficiency
    financial_ratio['AssetTurover'] = np.where(
        financial_data['totalAssets'] != 0,
        financial_data['totalRevenue'] / financial_data['totalAssets'],
        np.nan
    )
    financial_ratio['InventoryTurnover'] = np.where(
        financial_data['inventory'] != 0,
        financial_data['costofGoodsAndServicesSold'] / financial_data['inventory'],
        np.nan
    )
    financial_ratio['ARTurnover'] = np.where(
        financial_data['currentNetReceivables'] != 0,
        financial_data['totalRevenue'] / financial_data['currentNetReceivables'],
        np.nan
    )
    financial_ratio['APTurnover'] = np.where(
        financial_data['currentAccountsPayable'] != 0,
        financial_data['costofGoodsAndServicesSold'] / financial_data['currentAccountsPayable'],
        np.nan
    )

    # 4. Growth
    financial_ratio['RevenueGrowth'] = financial_data.groupby('comp')['totalRevenue'].pct_change()
    financial_ratio['NetIncomeGrowth'] = financial_data.groupby('comp')['netIncome'].pct_change()  
    financial_ratio['EPSGrowth'] = financial_data.groupby('comp')['reportedEPS'].pct_change()
    financial_ratio['FCFGrowth'] = financial_data.groupby('comp')['freeCashFlow'].pct_change()

    # 5. Valuation
    financial_ratio['DividendYield'] = np.where(
        financial_data['closePrice'] != 0,
        financial_data['dividend'] / financial_data['closePrice'],
        np.nan
    )
    financial_ratio['P/E'] = np.where(
        financial_data['reportedEPS'] != 0,
        financial_data['closePrice'] / financial_data['reportedEPS'],
        np.nan
    )
    financial_ratio['P/S'] = np.where(
        financial_data['totalRevenue'] != 0,
        financial_data['closePrice'] * financial_data['commonStockSharesOutstanding']  / financial_data['totalRevenue'],
        np.nan
    )
    financial_ratio['P/B'] = np.where(
        financial_data['totalShareholderEquity'] != 0,
        financial_data['closePrice'] * financial_data['commonStockSharesOutstanding']  / financial_data['totalShareholderEquity'],
        np.nan
    )
    # Cancel EV to EBITDA

    return financial_ratio

def process_technical_data(target: str):

    lookback_years = 10

    try:
        #end_date = dt.datetime.now()
        end_date = pd.to_datetime("2024/12/31")
        start_date = end_date - dt.timedelta(days=lookback_years*365)
        
        ticker = yf.Ticker(target)
        time.sleep(random.uniform(2, 5))
        stock_hist = ticker.history(start=start_date, end=end_date, interval='1d', auto_adjust=True)
        if stock_hist.empty:
            raise ValueError(f"No price history found for {target}. Ticker might be invalid or delisted.")
        
        stock_hist.index = stock_hist.index.tz_localize(None)
        if stock_hist.index[-1].date() == dt.date.today():
            stock_hist = stock_hist.iloc[:-1]
        if stock_hist.empty:
            raise ValueError(f"Data was found but removed (likely only 1 day available).")

        return stock_hist.sort_index(ascending=True)
    
    except Exception as e:
        raise ValueError(f"Technical Data Error for {target}: {str(e)}")
    
def calculate_technical_indicator(technical_data: pd.DataFrame):

    try:
        technical_indicator = technical_data[['Close', 'Volume']].copy()

        technical_indicator["RSI"] = ta.rsi(technical_data["Close"], length=14)

        macd= ta.macd(technical_data["Close"])
        technical_indicator['MACD_Line'] = macd['MACD_12_26_9']
        technical_indicator['MACD_Signal'] = macd['MACDs_12_26_9']
        technical_indicator['MACD_Hist'] = macd['MACDh_12_26_9']

        adx = ta.adx(
            high=technical_data["High"],
            low=technical_data["Low"],
            close=technical_data["Close"],
            length=14
        )
        technical_indicator['ADX'] = adx['ADX_14']
        technical_indicator['Trend_Dir'] = np.where(adx['DMP_14'] > adx['DMN_14'], "Bullish", "Bearish")

        technical_indicator["MFI"] = ta.mfi(
            high=technical_data["High"],
            low=technical_data["Low"],
            close=technical_data["Close"],
            volume=technical_data["Volume"],
            length=14
        )

        technical_indicator['MA_20'] = ta.sma(technical_data['Close'], length = 20)
        technical_indicator['MA_50'] = ta.sma(technical_data['Close'], length = 50)
        technical_indicator['MA_200'] = ta.sma(technical_data['Close'], length = 200)

        try:
            pivots = ta.pivots(technical_data['Open'], technical_data['High'], technical_data['Low'], technical_data['Close'], method='traditional', anchor='D')
            if not pivots.empty:
                if 'PIVOTS_TRAD_D_S1' in pivots.columns:
                    technical_indicator['Supp_1'] = pivots['PIVOTS_TRAD_D_S1']
                    technical_indicator['Res_1'] = pivots['PIVOTS_TRAD_D_R1']
                else:
                    technical_indicator['Supp_1'] = np.nan
                    technical_indicator['Res_1'] = np.nan
        except Exception:
            technical_indicator['Supp_1'] = np.nan
            technical_indicator['Res_1'] = np.nan

        analysis_days = 60
        technical_indicator = technical_indicator.tail(analysis_days)
        technical_indicator = technical_indicator.replace("nan", np.nan)
        technical_indicator = technical_indicator.dropna(axis=1, how='all')
        technical_indicator = technical_indicator.fillna("N/A")
        technical_indicator = technical_indicator.round(4)

        return technical_indicator
    
    except Exception as e:

        raise ValueError(f"Indicator Calculation Error: {str(e)}")


# --- 2. CREATING AI AGENT TOOLS & INPUTs ---

def make_competitors_tool(data: pd.DataFrame):

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

            financial_ratio_df = data
            financial_ratio_df = financial_ratio_df.sort_values(by='fiscalDateEnding', ascending=True)
            compare_df = financial_ratio_df.drop_duplicates(subset=['comp'], keep='last')

            compare_df = compare_df.replace("NaN", np.nan)
            compare_df = compare_df.dropna(axis=1, how='all')
            compare_df = compare_df.fillna("N/A")
            compare_df = compare_df.round(4)

            return {"status": "success", "comparison": compare_df.to_markdown(index=False)}

        except Exception as e:

            return {"status": "error", "error_message": str(e)}  
         
    return competitors_compare

def make_financial_ratio_tool(data: pd.DataFrame):

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

        target = target.upper()

        try:
            financial_ratio_df = data

            target_ratio_df = financial_ratio_df[financial_ratio_df['comp'] == target].copy()
            if target_ratio_df.empty:
                return {
                    "status": "error", 
                    "error_message": f"Stock ticker '{target}' not found in the database."
                }
            target_ratio_df['fiscalDateEnding'] = pd.to_datetime(target_ratio_df['fiscalDateEnding'])
            target_ratio_df = target_ratio_df.sort_values(by='fiscalDateEnding', ascending=True)
            #cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=5)
            cutoff_date = pd.to_datetime("2024/12/31") - pd.DateOffset(years=5)
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
        
    return get_financial_ratio

def fair_value_calculation(target: str, data: pd.DataFrame) -> str:
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
    
    #end_date = dt.datetime.now()
    end_date = pd.to_datetime("2024/12/31")
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
        financial_data_df = data
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
    
def make_technical_tool(data: pd.DataFrame):

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

        target = target.upper()

        try:

            technical_indicator = data

            return {
                "status": "success", 
                "technical_indicator": technical_indicator.to_markdown(index=True)
            }
        
        except Exception as e:

            return {"status": "error", "error_message": str(e)}
        
    return get_technical_analysis


# --- 3. DEFINE AGENT INITIALIZERS ---        

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

def competitors_agent_init(data):

    competitors_tool = make_competitors_tool(data)

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
    tools = [competitors_tool],
    output_key="comparing_competitors",
)

def financial_ratio_agent_init(data):

    financial_ratio_tool = make_financial_ratio_tool(data)

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
    tools = [financial_ratio_tool],
    output_key="financial_ratio_analysis",
)

def technical_agent_init(data):

    technical_tool = make_technical_tool(data)

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
    tools = [technical_tool],
    output_key="technical_analysis",
)

# --- 4. MAIN EXECUTION FLOW ---

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
    
    all_responses = []
    try:
        for event in response_generator:
            if event.content.parts and event.content.parts[0].text:
                all_responses.append(event.content.parts[0].text)
    except Exception as e:
        return f"Error reading generator: {e}"
    if all_responses:
        return all_responses[-1]
    
    return "Error: No output generated."

def run_stock_analysis(target, fundamental_data, financial_ratio, technical_indicator):

    news = news_agent_init()
    comp = competitors_agent_init(financial_ratio)
    fin_r = financial_ratio_agent_init(financial_ratio)
    tech = technical_agent_init(technical_indicator)
    
    dcf_result = fair_value_calculation(target, fundamental_data)
    
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
    - **Fair Price:** The valuation model result. **Market Price:** The latest market price for input data.
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
