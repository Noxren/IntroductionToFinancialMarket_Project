# IntroductionToFinancialMarket_Project
AI Agent in Asset Management

Data Source: 
yfinance <br />
https://ranaroussi.github.io/yfinance/reference/index.html <br />
Alpha Vantage <br />
https://www.alphavantage.co/documentation/ <br />
AI model: 
Google Gemini <br />
https://aistudio.google.com/ <br />
Note: Please don't share your API key 

## Work of the project
1. Download stock data using yfinance and Alpha Vantage.
2. Tidy up the data you need to input as a prompt to the AI Agent.
3. Build up your AI Agent with differnet instruction and prompt.
4. Check if the outcome is what you expect, if not, adjust your data or AI Agent.

----------------------------------------------
(After everyone complete their own AI Agent)

5. Discuss about what to include and what can be ignored in the final model.
6. Integrate the Agents and generate a research report.
7. Working on the group coursework report.
8. Prepare for the presentation.

## Basic Structure
![alt text](https://github.com/Noxren/IntroductionToFinancialMarket_Project/blob/main/Basic%20Strucutre.jpg?raw=true)

## How to implement
1. Download the "main" folder
2. Open 2 teminals in this folder
3. The 1st terminal: Type "uvicorn server:app" (backend)
4. The 2nd terminal: Type "streamlit run app.py" (fronTend)
