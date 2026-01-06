#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import sub_agents
import markdown
import pdfkit

def save_markdown_to_pdf(markdown_text, filename):
    
    # --- STEP 1: DEFINE THE PATH TO WKHTMLTOPDF ---
    # IMPORTANT: Use r'' (raw string) to handle backslashes correctly
    # If you installed it elsewhere, change this path!
    path_to_exe = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
    
    # Configure pdfkit to use this specific executable
    try:
        config = pdfkit.configuration(wkhtmltopdf=path_to_exe)
    except OSError:
        print("❌ CRITICAL ERROR: Could not find wkhtmltopdf.exe at the specified path.")
        print(f"Please check if this file exists: {path_to_exe}")
        return

    # --- STEP 2: CONVERT MARKDOWN TO HTML ---
    html_content = markdown.markdown(markdown_text, extensions=['tables'])

    styled_html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Helvetica', sans-serif; font-size: 12px; line-height: 1.6; color: #333; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div style="text-align:center; margin-bottom:30px;">
            <h1>Investment Memo: {target_ticker}</h1>
        </div>
        {html_content}
    </body>
    </html>
    """

    # --- STEP 3: GENERATE PDF USING CONFIG ---
    try:
        # NOTICE: We are passing 'configuration=config' here!
        pdfkit.from_string(styled_html, filename, configuration=config)
        print(f"✅ Professional PDF saved: {filename}")
    except Exception as e:
        print(f"❌ Error during PDF conversion: {e}")

if __name__ == "__main__":
    target_ticker = "SHEL"
    print(f"Generating Analysis for {target_ticker}...")
    final_memo = sub_agents.run_stock_analysis(target_ticker)

    filename = f"{target_ticker}_Report.pdf"
    save_markdown_to_pdf(final_memo, filename)
    print(f"✅ PDF successfully saved as: {filename}")

