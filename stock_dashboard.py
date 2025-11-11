import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import requests
import json
import time
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="Indian Stock Analysis Dashboard",
    page_icon="🇮🇳",
    layout="wide"
)

# --- Gemini API Configuration ---
# CRITICAL SECURITY: Load the key from secrets.
# Put your key in a file named .streamlit/secrets.toml
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

# Check if the key is loaded
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please add it to your .streamlit/secrets.toml file.")
    st.info("You can get a key from Google AI Studio: https://aistudio.google.com/app/apikey")
    st.stop() # Stop the app if the key is missing

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={GEMINI_API_KEY}"

# --- Helper Functions ---

def retry_with_backoff(api_call_function, max_retries=5, initial_delay=2):
    """Retries a function with exponential backoff."""
    delay = initial_delay
    for i in range(max_retries):
        try:
            return api_call_function()
        except requests.exceptions.RequestException as e:
            if i == max_retries - 1:
                st.error(f"API request failed after {max_retries} retries: {e}")
                return None
            # Show a friendlier, non-blocking warning for retries
            print(f"API request failed. Retrying in {delay}s... ({e})")
            time.sleep(delay)
            delay *= 2

def get_llm_sentiment_analysis(company_name, ticker):
    """
    Calls the Gemini API with Google Search grounding to get
    real-time news sentiment and analysis.
    """
    # This prompt asks for a balanced analysis (opportunities/risks)
    # and EXPLICITLY forbids investment advice.
    system_prompt = (
        "You are an expert financial analyst for the Indian stock market. "
        "Your purpose is to analyze real-time news for a given company. "
        "Based on the grounded Google Search results provided, you must: "
        "1. Provide a clear overall sentiment: **Positive**, **Negative**, or **Neutral**. "
        "2. Summarize the key **opportunities (bullish points)** based on the news. "
        "3. Summarize the key **risks (bearish points)** based on the news. "
        "4. Provide a final 2-3 sentence summary of the news. "
        "**IMPORTANT: You must not, under any circumstances, provide direct investment advice, price targets, or a 'buy'/'sell'/'hold' recommendation. Your role is to analyze the news, not to predict the stock price.**"
    )
    
    user_query = f"Analyze the latest news sentiment for {company_name} ({ticker})."

    payload = {
        "contents": [{ "parts": [{ "text": user_query }] }],
        "tools": [{ "google_search": {} }],
        "systemInstruction": {
            "parts": [{ "text": system_prompt }]
        },
    }

    headers = {'Content-Type': 'application/json'}

    def api_call():
        response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status() # Raises an HTTPError for bad responses
        return response.json()

    api_response = retry_with_backoff(api_call)

    if not api_response:
        return "Error: Could not fetch analysis from LLM.", []

    try:
        # Extract text
        text_response = api_response['candidates'][0]['content']['parts'][0]['text']
        
        # Extract sources
        sources = []
        grounding_metadata = api_response['candidates'][0].get('groundingMetadata', {})
        if 'groundingAttributions' in grounding_metadata:
            for attribution in grounding_metadata['groundingAttributions']:
                if 'web' in attribution and 'title' in attribution['web'] and 'uri' in attribution['web']:
                    sources.append({
                        "title": attribution['web']['title'],
                        "uri": attribution['web']['uri']
                    })
        
        return text_response, sources

    except (KeyError, IndexError, TypeError) as e:
        st.error(f"Error parsing LLM response: {e}")
        st.json(api_response) # Show the raw response for debugging
        return "Error: Could not parse analysis from LLM.", []


def fetch_stock_data(ticker_symbol):
    """Fetches stock data using yfinance."""
    ticker = yf.Ticker(ticker_symbol)
    
    # 1. Get 1 year of data for the main chart
    hist_1y = ticker.history(period="1y")
    
    # 2. Get 5 days of intraday data (15 min interval)
    hist_5d = ticker.history(period="5d", interval="15m")
    
    if hist_1y.empty or hist_5d.empty:
        return None, None, None
    
    # Get company info
    info = ticker.info
    
    return hist_1y, hist_5d, info

def plot_stock_chart(data, company_name):
    """Creates an interactive Plotly candlestick chart for 1 year."""
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=company_name
    )])

    fig.update_layout(
        title=f"{company_name} - 1 Year Stock Price",
        xaxis_title="Date",
        yaxis_title="Stock Price (INR)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0")
    )
    # --- BUG FIX ---
    # Corrected '1Errors' typo to '128'
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def plot_live_chart(data, company_name):
    """Creates an interactive Plotly line chart for 5 days."""
    fig = go.Figure(data=[go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name=company_name,
        line=dict(color='#00D0F0') # A nice cyan color for the line
    )])

    fig.update_layout(
        title=f"{company_name} - Live Chart (5-Day / 15-min Interval)",
        xaxis_title="Date / Time",
        yaxis_title="Stock Price (INR)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0")
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

# --- Streamlit App UI ---

st.title("Indian Stock Analysis Dashboard (By Shivam Rawat)")
st.markdown("Enter an NSE/BSE stock ticker (e.g., `RELIANCE.NS`, `TCS.NS`, `INFY.BO`) to get data and AI-powered news sentiment.")

# Input section
col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input("Enter Stock Ticker:", "RELIANCE.NS").upper()
with col2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True) # Spacer
    analyze_button = st.button("Analyze Stock", type="primary", use_container_width=True)

# Analysis output section
if analyze_button:
    if not ticker_input:
        st.warning("Please enter a stock ticker.")
    else:
        with st.spinner(f"Analyzing {ticker_input}... Fetching data, graphs, and LLM analysis..."):
            try:
                # 1. Fetch Stock Data
                stock_data_1y, stock_data_5d, company_info = fetch_stock_data(ticker_input)

                if stock_data_1y is None or stock_data_5d is None or company_info is None:
                    st.error(f"Could not find data for ticker: {ticker_input}. Is it a valid Indian stock ticker (e.g., 'RELIANCE.NS')?")
                else:
                    company_name = company_info.get('longName', ticker_input)
                    
                    # 2. Display Header & Info
                    st.header(f"{company_name} ({company_info.get('symbol', '')})")
                    
                    # Key metrics - Changed to 2x2 grid for better mobile layout
                    col1, col2 = st.columns(2)
                    col3, col4 = st.columns(2)
                    
                    current_price = company_info.get('currentPrice', 'N/A')
                    if current_price != 'N/A':
                        day_high = company_info.get('dayHigh', 'N/A')
                        day_low = company_info.get('dayLow', 'N/A')
                        change = company_info.get('regularMarketChange', 0)
                        change_pct = company_info.get('regularMarketChangePercent', 0)
                        
                        delta = f"{change:.2f} ({change_pct*100:.2f}%)"
                        
                        # --- BUG FIX ---
                        # Removed the invalid `delta_color` argument.
                        col1.metric("Current Price", f"₹{current_price}", delta=delta)
                        # -----------------
                        
                        col2.metric("Day High", f"₹{day_high}")
                        col3.metric("Day Low", f"₹{day_low}")
                    else:
                        # Fallback for when 'currentPrice' isn't available
                        last_close = company_info.get('previousClose', 'N/A')
                        col1.metric("Last Close", f"₹{last_close}")
                        col3.metric("Market Cap", f"{company_info.get('marketCap', 'N/A'):,}")
                    
                    col4.metric("52 Week High", f"₹{company_info.get('fiftyTwoWeekHigh', 'N/A')}")

                    st.markdown("---")

                    # 3. Create Tabs
                    # Added a new tab for the Live Chart
                    tab1, tab2, tab3, tab4 = st.tabs(["📊 Live (5-Day)", "📈 1-Year Chart", "🤖 LLM News Analysis", "ℹ️ Company Info"])
                    
                    with tab1:
                        # Plot Live Chart
                        st.plotly_chart(plot_live_chart(stock_data_5d, company_name), use_container_width=True)
                    
                    with tab2:
                        # Plot 1-Year Chart
                        st.plotly_chart(plot_stock_chart(stock_data_1y, company_name), use_container_width=True)
                    
                    with tab3:
                        # 4. Get LLM Sentiment
                        st.subheader("🤖 Real-Time News Analysis")
                        st.markdown(f"Powered by Gemini 2.5 Flash with Google Search grounding.")
                        
                        analysis_text, news_sources = get_llm_sentiment_analysis(company_name, ticker_input)
                        
                        # Display sentiment analysis
                        st.markdown(analysis_text)
                        
                        # Display sources
                        if news_sources:
                            with st.expander("See News Sources"):
                                for i, source in enumerate(news_sources):
                                    st.markdown(f"[{i+1}. {source['title']}]({source['uri']})")
                    
                    with tab4:
                        # Display Company Info
                        st.subheader("Company Profile")
                        st.markdown(f"**Sector:** {company_info.get('sector', 'N/A')}")
                        st.markdown(f"**Industry:** {company_info.get('industry', 'N/A')}")
                        st.markdown(f"**Website:** {company_info.get('website', 'N/A')}")
                        st.markdown(f"**Location:** {company_info.get('city', 'N/A')}, {company_info.get('country', 'N/A')}")
                        
                        st.subheader("Business Summary")
                        st.info(company_info.get('longBusinessSummary', 'No summary available.'))

            except Exception as e:
                # Use st.exception to print the full traceback for easier debugging
                st.exception(f"An unexpected error occurred: {e}")