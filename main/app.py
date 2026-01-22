import streamlit as st
import requests
import pandas as pd
import altair as alt

# --- 頁面設定 ---
st.set_page_config(layout="wide", page_title="Stock AI Agent")

# --- Session State 初始化 ---
keys_to_init = {
    'active_symbol': None,
    'ticker_input': "",
    'search_results': [],
    'auto_run_analysis': False,
    'last_search': "" # Added this to prevent logic errors
}

for key, default in keys_to_init.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- 設定後端 URL ---
BACKEND_URL = "http://localhost:8000"
session = requests.Session()
session.trust_env = False 

# --- 輔助函數 ---
def set_ticker(symbol):
    st.session_state.ticker_input = symbol  
    st.session_state.active_symbol = symbol
    st.session_state.auto_run_analysis = True     
    st.session_state.search_results = []
    st.session_state.last_search = symbol

# --- 標題 ---
st.markdown("""
<h1 style='text-align: center; color: var(--primary-color);'>
    Stock Analysis Agent
</h1>
<p style='text-align: center; color: var(--text-color); font-size: 24px;'>
    Powered by AI-driven financial insights
</p>
<hr>
""", unsafe_allow_html=True)

# --- Sidebar: 搜尋與設定 ---
with st.sidebar:
    st.header("Control Panel")

    # 搜尋框
    current_input = st.text_input("Enter Ticker (e.g., SHEL, NVDA):", value=st.session_state.ticker_input)
    is_new_term = current_input and current_input != st.session_state.last_search
    is_active_symbol = current_input == st.session_state.active_symbol

    # 搜尋邏輯
    search_triggered = st.button("Start Searching", type="primary") or (is_new_term and not is_active_symbol)

    if search_triggered:
        st.session_state.last_search = current_input

        if current_input:
            with st.spinner(f"Searching for '{current_input}'..."):
                try:
                    response = session.get(f"{BACKEND_URL}/api/search", params={"keyword": current_input})
                    if response.status_code == 200:
                        st.session_state.search_results = response.json().get("data", [])
                        if not st.session_state.search_results:
                            st.warning("No results found.")
                    else:
                        st.error("API Error during search.")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

    # 顯示搜尋結果 (🔥 修正錯誤的地方)
    if st.session_state.search_results:
        st.write("---") 
        st.write("### Search Results:")
        # 使用 enumerate 取得 index 'i'，確保 key 是唯一的
        for i, item in enumerate(st.session_state.search_results):
            col1, col2 = st.columns([2, 1], vertical_alignment="center")
            with col1:
                st.markdown(f"**{item['symbol']}**") 
                st.caption(f"{item['name']}")
            with col2:
                # 將 index 加入 key 中：key=f"sel_{item['symbol']}_{i}"
                if st.button("Select", key=f"sel_{item['symbol']}_{i}"):
                    st.session_state.search_results = []
                    set_ticker(item['symbol'])
                    st.rerun()
            st.markdown("---")

# --- 主畫面邏輯 ---
if st.session_state.active_symbol:
    ticker = st.session_state.active_symbol
    st.markdown(f"## Analyzing: **{ticker}**")

    # 定義分頁 (已移除 Technical Agent)
    tab1, tab2, tab3, tab4 = st.tabs([
        "Investment Memo", 
        "Fundamental Analysis", 
        "Technical Analysis", 
        "Backtesting"
    ])

    # === Tab 1: AI Investment Memo ===
    with tab1:
        st.subheader("AI-Generated Investment Report")
        
        def trigger_analysis():
            with st.spinner("AI Agents are researching news, financials, and competitors... (This may take few minutes)"):
                try:
                    response = session.post(f"{BACKEND_URL}/api/analyze_ai/{ticker}")
                    if response.status_code == 200:
                        st.success("Analysis Complete!")
                        st.session_state.search_results = [] 
                        st.session_state.auto_run_analysis = False
                        st.rerun()
                    else:
                        st.error(f"Analysis failed: {response.text}")
                        st.session_state.auto_run_analysis = False
                except Exception as e:
                    st.error(f"Connection Error: {e}")
                    st.session_state.auto_run_analysis = False

        col_btn, col_status = st.columns([1, 4])

        with col_btn:
            if st.button(f"Generate New Report", type="primary", use_container_width=True):
                trigger_analysis()
                
        if st.session_state.get('auto_run_analysis', False):
            trigger_analysis()

        st.markdown("---")
        
        try:
            report_res = session.get(f"{BACKEND_URL}/api/get_ai_report/{ticker}")
            
            if report_res.status_code == 200:
                report_data = report_res.json()
                report_date = report_data.get("date", "Unknown Date")
                memo_content = report_data.get("analysis", "No content available.")
                
                d_col1, d_col2 = st.columns([3, 1])
                with d_col1:
                    st.info(f"Report Generated on: **{report_date}**")
                with d_col2:
                    # Add a download button for the text
                    st.download_button(
                        label="Download Report",
                        data=memo_content,
                        file_name=f"{ticker}_Investment_Memo_{report_date}.md",
                        mime="text/markdown"
                    )
                with st.container(border=True):
                    memo_content = memo_content.replace("$", r"\$")
                    st.markdown(memo_content)
            else:
                st.warning("No report found for this session.")
                st.markdown(f"Click **Generate New Report** to let the AI analyze **{ticker}**.")
        
        except Exception as e:
            st.error(f"Could not retrieve report: {e}")

    # === Tab 2: Financial Stats ===
    with tab2:
        st.subheader("Fundamental Analysis")
        try:
            fundamental_res = session.get(f"{BACKEND_URL}/api/fundamental/{ticker}")
            if fundamental_res.status_code == 200:
                fundamental_dict = fundamental_res.json()
                report_date = fundamental_dict.get("date", "Unknown Date")
                fundamental_data = fundamental_dict.get("financial_ratio", [])
                
                st.info(f"Report Generated on: {report_date}")

                if fundamental_data:
                    fundamental_data = pd.DataFrame(fundamental_data)
                    fundamental_data['fiscalDateEnding'] = pd.to_datetime(fundamental_data['fiscalDateEnding'])
                    fundamental_data['year'] = fundamental_data['fiscalDateEnding'].dt.year

                    target_data = fundamental_data[fundamental_data['comp'].str.upper() == ticker.upper()].copy()
                    competitors_data = fundamental_data[fundamental_data['comp'].str.upper() != ticker.upper()].copy()
                    numeric_cols = competitors_data.select_dtypes(include=['number']).columns
                    numeric_cols = [c for c in numeric_cols if c != 'year']
                    competitors_avg = competitors_data.groupby('year')[numeric_cols].mean().reset_index()
                    competitors_avg['comp'] = 'competitorsAvg'
                    plot_data = pd.concat([target_data, competitors_avg], ignore_index=True)

                    subtab_profitabilty, subtab_leverage, subtab_efficiency, subtab_growth, subtab_valuation = st.tabs(["Profitability", "Leverage", "Efficiency", "Growth", "Valuation"])
                    
                    def make_chart(data, y_col, title, format=".2f"):
                        base = alt.Chart(data).encode(
                            x=alt.X('year:O', title='Year'),
                            y=alt.Y(y_col, title=title),
                            tooltip=['comp', 'year', alt.Tooltip(y_col, format=format)]
                        )

                        target_line = base.transform_filter(
                            alt.datum.comp != 'competitorsAvg'
                        ).mark_line(point=True, strokeWidth=3).encode(
                            color=alt.value('#2962FF') # Bright Blue
                        )

                        avg_line = base.transform_filter(
                            alt.datum.comp == 'competitorsAvg'
                        ).mark_line(point=True, strokeDash=[5,5], strokeWidth=2).encode(
                            color=alt.value('gray')
                        )

                        return (target_line + avg_line).properties(height=300).interactive()
                    
                    with subtab_profitabilty:
                        st.markdown("#### Margins & Returns")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Gross & Net Margins**")
                            st.altair_chart(make_chart(plot_data, 'GrossMargin', 'Gross Margin', '.1%'), use_container_width=True)
                            st.altair_chart(make_chart(plot_data, 'NetProfitMargin', 'Net Margin', '.1%'), use_container_width=True)
                        with col2:
                            st.markdown("**Return on Equity & Assets**")
                            st.altair_chart(make_chart(plot_data, 'ROE', 'ROE', '.1%'), use_container_width=True)
                            st.altair_chart(make_chart(plot_data, 'ROA', 'ROA', '.1%'), use_container_width=True)
                    
                    with subtab_leverage:
                        st.markdown("#### Debt & Liquidity")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Debt to Equity (D/E)**")
                            st.caption("Lower is generally better.")
                            st.altair_chart(make_chart(plot_data, 'D/E', 'Debt/Equity Ratio'), use_container_width=True)
                        
                            st.markdown("**Debt to EBITDA**")
                            st.altair_chart(make_chart(plot_data, 'Debt/EBITDA', 'Debt/EBITDA'), use_container_width=True)
                        with col2:
                            st.markdown("**Current Ratio**")
                            st.caption("Above 1.0 indicates good short-term liquidity.")
                            st.altair_chart(make_chart(plot_data, 'CurrentRatio', 'Current Ratio'), use_container_width=True)
                        
                            st.markdown("**Interest Coverage**")
                            st.caption("Ability to pay interest on outstanding debt.")
                            st.altair_chart(make_chart(plot_data, 'InterestCoverage', 'Interest Coverage'), use_container_width=True)
                    
                    with subtab_efficiency:
                        st.markdown("#### Operational Efficiency")
                    
                        st.markdown("**Inventory Turnover**")
                        st.altair_chart(make_chart(plot_data, 'InventoryTurnover', 'Inventory Turnover'), use_container_width=True)
                    
                        col1, col2 = st.columns(2)
                        with col1:
                            st.altair_chart(make_chart(plot_data, 'AssetTurover', 'Asset Turnover'), use_container_width=True)
                        with col2:
                            st.altair_chart(make_chart(plot_data, 'ARTurnover', 'Receivables Turnover'), use_container_width=True)

                    with subtab_growth:
                        st.markdown("#### Growth Rates (YoY)")
                        st.caption("Percentage change from previous year.")
                    
                        col1, col2 = st.columns(2)
                        with col1:
                            st.altair_chart(make_chart(plot_data, 'RevenueGrowth', 'Revenue Growth', '.1%'), use_container_width=True)
                            st.altair_chart(make_chart(plot_data, 'EPSGrowth', 'EPS Growth', '.1%'), use_container_width=True)
                        with col2:
                            st.altair_chart(make_chart(plot_data, 'NetIncomeGrowth', 'Net Income Growth', '.1%'), use_container_width=True)
                            st.altair_chart(make_chart(plot_data, 'FCFGrowth', 'Free Cash Flow Growth', '.1%'), use_container_width=True)

                    with subtab_valuation:
                        st.markdown("#### Valuation Multiples")
                        col1, col2 = st.columns(2)
                    
                        with col1:
                            st.altair_chart(make_chart(plot_data, 'P/E', 'Price to Earnings (P/E)'), use_container_width=True)
                            st.altair_chart(make_chart(plot_data, 'P/S', 'Price to Sales (P/S)'), use_container_width=True)
                    
                        with col2:
                            st.altair_chart(make_chart(plot_data, 'P/B', 'Price to Book (P/B)'), use_container_width=True)
                            st.altair_chart(make_chart(plot_data, 'DividendYield', 'Dividend Yield', '.2%'), use_container_width=True)
                    
                    st.caption("🔵 **Blue Line:** Target Company | ⚪ **Gray Dashed:** Industry Average")
                else:
                    st.warning("No financial ratio data available.")
            else:
                st.error("Failed to fetch fundamental data.")
        except Exception as e:
            st.error(f"Error: {e}")

    with tab3:
        st.subheader("Technical Analysis")
        try:
            technical_res = session.get(f"{BACKEND_URL}/api/technical/{ticker}")
            if technical_res.status_code == 200:
                technical_dict = technical_res.json()
                report_date = technical_dict.get("date", "Unknown Date")
                technical_data = technical_dict.get("technical_indicator", [])

                if technical_data:
                    technical_data = pd.DataFrame(technical_data)
                    technical_data.reset_index()
                    technical_data['Date'] = pd.to_datetime(technical_data['Date'])

                    st.info(f"Report Generated on: {report_date}")

                    subtab_price, subtab_momentum, subtab_trend = st.tabs(["Price & MAs", "Momentum", "Trend"])
                    with subtab_price:
                        st.markdown("##### Price vs Moving Averages")
        
                        # Base Chart
                        base = alt.Chart(technical_data).encode(x='Date:T')
        
                        # 1. Close Price Line
                        price_line = base.mark_line(color='black', strokeWidth=2).encode(
                            y=alt.Y('Close', scale=alt.Scale(zero=False), title='Price'),
                            tooltip=['Date', 'Close', 'Volume']
                        )
        
                        # 2. Moving Averages (Using different colors)
                        ma_20 = base.mark_line(color='#FFD700', strokeDash=[5,5]).encode(y='MA_20', tooltip=['MA_20']) # Gold
                        ma_50 = base.mark_line(color='#2196F3').encode(y='MA_50', tooltip=['MA_50'])  # Blue
                        ma_200 = base.mark_line(color='#F44336').encode(y='MA_200', tooltip=['MA_200']) # Red
        
                        # 3. Support & Resistance (Points or Step Lines)
                        # We use circles to show the S1/R1 levels for each day
                        supp_1 = base.mark_circle(color='green', size=20).encode(y='Supp_1', tooltip=['Supp_1'])
                        res_1 = base.mark_circle(color='red', size=20).encode(y='Res_1', tooltip=['Res_1'])

                        # Combine
                        chart1 = (price_line + ma_20 + ma_50 + ma_200 + supp_1 + res_1).properties(height=400).interactive()
                        st.altair_chart(chart1, use_container_width=True)
        
                        # Legend/Key
                        st.caption("🟡 MA-20 (Short) | 🔵 MA-50 (Medium) | 🔴 MA-200 (Long) | Dots: Support/Resistance")

                    with subtab_momentum:
        
                        # --- RSI Chart ---
                        st.markdown("##### RSI (Relative Strength Index)")
                        base_rsi = alt.Chart(technical_data).encode(x='Date:T')
        
                        rsi_line = base_rsi.mark_line(color='purple').encode(
                            y=alt.Y('RSI', scale=alt.Scale(domain=[0, 100]))
                        )
        
                        # Add 30/70 Reference Lines
                        rules_df = pd.DataFrame({'y': [30, 70], 'Label': ['Oversold', 'Overbought']})
                        rules = alt.Chart(rules_df).mark_rule(color='gray', strokeDash=[3,3]).encode(y='y')
        
                        st.altair_chart((rsi_line + rules).properties(height=250), use_container_width=True)

                        # --- MFI Chart ---
                        st.markdown("##### MFI (Money Flow Index)")
                        mfi_line = base_rsi.mark_line(color='teal').encode(
                            y=alt.Y('MFI', scale=alt.Scale(domain=[0, 100]))
                        )
                        # MFI usually uses 20/80 as levels
                        rules_mfi_df = pd.DataFrame({'y': [20, 80]})
                        rules_mfi = alt.Chart(rules_mfi_df).mark_rule(color='gray', strokeDash=[3,3]).encode(y='y')
        
                        st.altair_chart((mfi_line + rules_mfi).properties(height=250), use_container_width=True)

                    with subtab_trend:
        
                        # --- MACD Chart ---
                        st.markdown("##### MACD")
                        base_macd = alt.Chart(technical_data).encode(x='Date:T')
        
                        # The Histogram (Green/Red bars)
                        hist = base_macd.mark_bar().encode(
                            y='MACD_Hist',
                            color=alt.condition(
                                alt.datum.MACD_Hist > 0,
                                alt.value("green"),
                                alt.value("red")
                            )
                        )
        
                        # The Lines
                        macd_line = base_macd.mark_line(color='blue').encode(y='MACD_Line')
                        signal_line = base_macd.mark_line(color='orange').encode(y='MACD_Signal')
        
                        st.altair_chart((hist + macd_line + signal_line).properties(height=300), use_container_width=True)

                        # --- ADX Chart ---
                        st.markdown("##### ADX (Trend Strength)")
        
                        # We color the ADX line based on the Trend Direction you calculated!
                        adx_chart = alt.Chart(technical_data).mark_line(strokeWidth=3).encode(
                            x='Date:T',
                            y='ADX',
                            color=alt.Color('Trend_Dir', scale=alt.Scale(domain=['Bullish', 'Bearish'], range=['green', 'red'])),
                            tooltip=['Date', 'ADX', 'Trend_Dir']
                        ).properties(height=250)
        
                        st.altair_chart(adx_chart, use_container_width=True)
                        st.caption("Note: High ADX (>25) indicates a strong trend. The color indicates if that trend is Bullish or Bearish.")
                else:
                    st.warning("No financial ratio data available.")
            else:
                st.error("Failed to fetch technical data.")

        except Exception as e:
            st.error(f"Error: {e}")


    # === Tab 4: Backtesting ===
    with tab4:
        st.subheader("Simple Backtest Strategy")
        st.write("This feature is currently under development.")

else:
    st.info("👈 Please search for a stock symbol in the sidebar to begin.")