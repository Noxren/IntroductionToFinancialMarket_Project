import streamlit as st
import requests
import pandas as pd
import altair as alt
import datetime as dt

st.set_page_config(layout="wide", page_title="Stock AI Agent")

# --- Session State Initialization ---
keys_to_init = {
    'active_symbol': None,
    'ticker_input': "",
    'search_results': [],
    'auto_run_analysis': False,
    'last_search': ""
}

for key, default in keys_to_init.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Setting Backend URL ---
BACKEND_URL = "http://localhost:8000"
session = requests.Session()
session.trust_env = False 

# --- Support Function ---
def set_ticker(symbol):
    st.session_state.ticker_input = symbol  
    st.session_state.active_symbol = symbol
    st.session_state.auto_run_analysis = True     
    st.session_state.search_results = []
    st.session_state.last_search = symbol
    st.session_state.terminal_growth_rate = 2.0
    st.session_state.date = dt.date.today().strftime("%Y-%m-%d")
    st.session_state.backtest_result = None

# --- Title ---
st.markdown("""
<h1 style='text-align: center; color: var(--primary-color);'>
    Stock Analysis Agent
</h1>
<p style='text-align: center; color: var(--text-color); font-size: 24px;'>
    Powered by AI-driven financial insights
</p>
<hr>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("Control Panel")

    if not st.session_state.active_symbol:
        current_input = st.text_input("Enter Ticker (e.g., SHEL, NVDA):", value=st.session_state.ticker_input)
        is_new_term = current_input and current_input != st.session_state.last_search

        search_triggered = st.button("Start Searching", type="primary", use_container_width=True) or is_new_term

        if search_triggered:
            st.session_state.last_search = current_input
            st.session_state.search_results = []

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

        if st.session_state.search_results:
            st.markdown("---")
            st.caption("Select a company:")
            for i, item in enumerate(st.session_state.search_results):
                with st.container(border=True):
                    col1, col2 = st.columns([2, 1], vertical_alignment="center")
                    with col1:
                        st.markdown(f"**{item['symbol']}**") 
                        st.caption(f"{item['name']}")
                    with col2:
                        if st.button("Select", key=f"sel_{item['symbol']}_{i}"):
                            set_ticker(item['symbol'])
                            st.session_state.auto_run_analysis = False
                            st.session_state.search_results = []
                            st.rerun()

    else:
        st.subheader("Valuation Settings")
        st.success(f"Active Stock: **{st.session_state.active_symbol}**")
        st.markdown("Set Parameters:")
        terminal_growth_input = st.number_input(
            "Terminal Growth Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            format="%.1f",
            help="Perpetual growth rate for DCF Terminal Value."
        )
        st.session_state.terminal_growth_rate = terminal_growth_input
        st.markdown("---")
        if st.button("Clear Selection", type="secondary", use_container_width=True):
            set_ticker("") 
            st.session_state.search_results = []
            st.rerun()

# --- Main Page Logic ---
if st.session_state.active_symbol:
    ticker = st.session_state.active_symbol
    st.markdown(f"## Analyzing: **{ticker}**")

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
            terminal_growth_rate = st.session_state.get("terminal_growth_rate") / 100.0
            date = st.session_state.get("date")
            with st.status("AI Agents working... (This may take a few minutes)", expanded=True) as status:
                try:
                    response = session.post(f"{BACKEND_URL}/api/analyze_ai/{ticker}",
                                            params={"date": date, "terminal_growth_rate": terminal_growth_rate})
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
            if st.button(f"Run Analysis for {ticker}", type="primary", use_container_width=True):
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
        
        except Exception as e:
            st.error(f"Could not retrieve report: {e}")

    # === Tab 2: Fundamental Analysis ===
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

    # === Tab3: Technical Analysis ===
    with tab3:
        st.subheader("Technical Analysis")
        try:
            technical_res = session.get(f"{BACKEND_URL}/api/technical/{ticker}")
            if technical_res.status_code == 200:
                technical_dict = technical_res.json()
                report_date = technical_dict.get("date", "Unknown Date")
                price_data = technical_dict.get("price_data", [])
                technical_data = technical_dict.get("technical_indicator", [])

                if price_data and technical_data :
                    price_data = pd.DataFrame(price_data)
                    price_data.reset_index()
                    price_data['Date'] = pd.to_datetime(price_data['Date'])

                    technical_data = pd.DataFrame(technical_data)
                    technical_data.reset_index()
                    technical_data['Date'] = pd.to_datetime(technical_data['Date'])

                    cols_to_merge = [col for col in technical_data.columns if col not in price_data.columns or col == 'Date']
                    combined_df = pd.merge(price_data, technical_data[cols_to_merge], on='Date', how='inner')
                    combined_df = combined_df.sort_values('Date')
                    combined_df = combined_df.reset_index(drop=True)
                    combined_df['Candle_ID'] = combined_df.index
                    combined_df['DateStr'] = combined_df['Date'].astype(str)

                    st.info(f"Report Generated on: {report_date}")

                    subtab_price, subtab_momentum, subtab_trend = st.tabs(["Price & MA", "Momentum", "Trend"])
                    
                    with subtab_price:
                        st.markdown("##### Price vs Moving Averages")
        
                        brush = alt.selection_interval(encodings=['x'])
                        base = alt.Chart(combined_df).encode(
                            x=alt.X('Candle_ID:Q', axis=alt.Axis(
                                title='Date',
                                labels=False
                            ))
                        ).properties(width='container')
        
                        # 1. Candlelight Chart
                        # Part A: The vertical line (High to Low)
                        rule = base.mark_rule().encode(
                            y=alt.Y('High:Q', scale=alt.Scale(zero=False), title='Price'),
                            y2='Low:Q',
                            color=alt.condition("datum.Open <= datum.Close", alt.value("#00C805"), alt.value("#FF333A"))
                        )

                        # Part B: The body (Open to Close)
                        bar = base.mark_bar().encode(
                            y='Open:Q',
                            y2='Close:Q',
                            color=alt.condition("datum.Open <= datum.Close", alt.value("#00C805"), alt.value("#FF333A")),
                            tooltip=[
                                alt.Tooltip('DateStr', title='Date'),
                                alt.Tooltip('Open', title='Open', format=",.2f"),
                                alt.Tooltip('High', title='High', format=",.2f"),
                                alt.Tooltip('Low', title='Low', format=",.2f"),
                                alt.Tooltip('Close', title='Close', format=",.2f"),
                                alt.Tooltip('Volume', title='Volume', format=",.0f")
                            ]
                        )   
                        layers = [rule, bar]

                        # 2. Moving Averages (Using different colors)
                        layers.append(base.mark_line(color='#FFD700', strokeDash=[5,5]).encode(y='MA_20', tooltip=['MA_20']))
                        layers.append(base.mark_line(color='#2196F3').encode(y='MA_50', tooltip=['MA_50'])) 
                        layers.append(base.mark_line(color='#F44336').encode(y='MA_200', tooltip=['MA_200']))
        
                        price_chart = alt.layer(*layers).properties(height=400)
                        
                        # 3. Volume (Subplot)
                        volume_chart = base.mark_bar().encode(
                            y=alt.Y('Volume:Q', title='Vol'),
                            color=alt.condition("datum.Open <= datum.Close", alt.value("#00C805"), alt.value("#FF333A")),
                            tooltip=[alt.Tooltip('DateStr', title='Date'), alt.Tooltip('Volume', title='Volume', format=",.0f")]
                        ).properties(height=100)

                        final_chart = alt.vconcat(price_chart, volume_chart, spacing=5).resolve_scale(x='shared')

                        st.altair_chart(final_chart, use_container_width=True)
        
                        st.caption("🟢/🔴 Candles: Price Action | 🟡 MA-20 | 🔵 MA-50 | 🔴 MA-200 | Bar: Volume")

                    with subtab_momentum:
        
                        # --- RSI Chart ---
                        st.markdown("##### RSI (Relative Strength Index)")
                        base_rsi = alt.Chart(technical_data).encode(x='Date:T')
        
                        rsi_line = base_rsi.mark_line(color='purple').encode(
                            y=alt.Y('RSI', scale=alt.Scale(domain=[0, 100]))
                        )
        
                        rules_df = pd.DataFrame({'y': [30, 70], 'Label': ['Oversold', 'Overbought']})
                        rules = alt.Chart(rules_df).mark_rule(color='gray', strokeDash=[3,3]).encode(y='y')
        
                        st.altair_chart((rsi_line + rules).properties(height=250), use_container_width=True)

                        # --- MFI Chart ---
                        st.markdown("##### MFI (Money Flow Index)")
                        mfi_line = base_rsi.mark_line(color='teal').encode(
                            y=alt.Y('MFI', scale=alt.Scale(domain=[0, 100]))
                        )
                        rules_mfi_df = pd.DataFrame({'y': [20, 80]})
                        rules_mfi = alt.Chart(rules_mfi_df).mark_rule(color='gray', strokeDash=[3,3]).encode(y='y')
        
                        st.altair_chart((mfi_line + rules_mfi).properties(height=250), use_container_width=True)

                    with subtab_trend:
        
                        # --- MACD Chart ---
                        st.markdown("##### MACD")
                        base_macd = alt.Chart(technical_data).encode(x='Date:T')
        
                        hist = base_macd.mark_bar().encode(
                            y='MACD_Hist',
                            color=alt.condition(
                                alt.datum.MACD_Hist > 0,
                                alt.value("green"),
                                alt.value("red")
                            )
                        )
        
                        macd_line = base_macd.mark_line(color='blue').encode(y='MACD_Line')
                        signal_line = base_macd.mark_line(color='orange').encode(y='MACD_Signal')
        
                        st.altair_chart((hist + macd_line + signal_line).properties(height=300), use_container_width=True)

                        # --- ADX Chart ---
                        st.markdown("##### ADX (Trend Strength)")
        
                        adx_chart = alt.Chart(technical_data).mark_line(strokeWidth=3).encode(
                            x='Date:T',
                            y='ADX',
                            color=alt.Color('Trend_Dir', scale=alt.Scale(domain=['Bullish', 'Bearish'], range=['green', 'red'])),
                            tooltip=['Date', 'ADX', 'Trend_Dir']
                        ).properties(height=250)
        
                        st.altair_chart(adx_chart, use_container_width=True)
                        st.caption("Note: High ADX (>25) indicates a strong trend. The color indicates if that trend is Bullish or Bearish.")
                
                        # --- Support & Resistance ---
                        st.markdown("##### Key Levels (Support & Resistance)")
                    
                        base_trend = alt.Chart(combined_df).encode(
                            x=alt.X('Candle_ID:Q', axis=alt.Axis(title='Date', labels=False))
                        ).properties(width='container')

                        line = base_trend.mark_line(color='blue', strokeWidth=2).encode(
                            y=alt.Y('Close:Q', scale=alt.Scale(zero=False), title='Price'),
                            tooltip=[alt.Tooltip('DateStr', title='Date'), alt.Tooltip('Close', title='Close', format=",.2f")]
                        )

                        layers = [line]
                    
                        layers.append(base_trend.mark_line(color='red', strokeDash=[4,2], interpolate='step-after').encode(
                            y='Res_1:Q',
                            tooltip=[alt.Tooltip('Res_1', title='Resistance', format=",.2f")]
                        ))
                        layers.append(base_trend.mark_text(align='left', dx=5, color='red').encode(
                            x=alt.value(400), 
                            y=alt.Y('Res_1:Q', aggregate='max'), 
                            text=alt.value("Resistance")
                        ))

                        layers.append(base_trend.mark_line(color='green', strokeDash=[4,2], interpolate='step-after').encode(
                            y='Supp_1:Q',
                            tooltip=[alt.Tooltip('Supp_1', title='Support', format=",.2f")]
                        ))
                        layers.append(base_trend.mark_text(align='left', dx=5, color='green').encode(
                            x=alt.value(400),
                            y=alt.Y('Supp_1:Q', aggregate='max'),
                            text=alt.value("Support")
                        ))

                        st.altair_chart(alt.layer(*layers).properties(height=350), use_container_width=True)
                else:
                    st.warning("No financial ratio data available.")
            else:
                st.error("Failed to fetch technical data.")

        except Exception as e:
            st.error(f"Error: {e}")


    # === Tab 4: Backtesting ===
    with tab4:
        st.subheader("Backtesting")
        st.write("Test how accurate the AI agent is base on historical date.")

        def clear_backtest_result():
                if st.session_state.get('backtest_result'):
                    st.session_state.backtest_result = None

        col1, col2 = st.columns([2, 1])
        with col1:
            backtest_date = st.date_input(
                "Select Backtest Date",
                value=dt.date(2024, 12, 31),
                min_value=pd.to_datetime("2010-01-01"),
                max_value=dt.date.today() - dt.timedelta(days=30),
                on_change=clear_backtest_result,
                help="The model will ignore all data after this date."
            )
        with col2:
            st.write("")  
            st.write("") 
            run_backtest = st.button("Run Backtest", type="primary", use_container_width=True)
        
        if run_backtest:
            date = backtest_date.strftime("%Y-%m-%d")
            terminal_growth_rate = st.session_state.get('terminal_growth_rate', 2.0) / 100.0
            with st.spinner(f"Running backtest for {ticker} as of {date}..."):
                try:
                    response = session.get(
                        f"{BACKEND_URL}/api/backtest/{ticker}",
                        params={"date": date, "terminal_growth_rate":terminal_growth_rate}
                    )
                    if response.status_code == 200:
                        report_res = session.get(f"{BACKEND_URL}/api/get_backtest_report/{ticker}",
                                                 params={"date": date})
                        if report_res.status_code == 200:
                            st.session_state.backtest_result = report_res.json()
                            st.success("Backtest Complete!")
                        else:
                            st.error("Analysis finished, but could not retrieve report.")
                    else:
                        st.error(f"Analysis failed: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")
        
        if st.session_state.get('backtest_result'):
            bt_data = st.session_state.backtest_result
        
            st.divider()
            st.subheader(f"Backtest Results: {bt_data.get('date', 'Unknown Date')}")
        
            content = bt_data.get("analysis", "No content.")
            content = content.replace("$", r"\$") 
        
            with st.container(border=True):
                st.markdown(content)
