#Shahab Ebadi
#Asiatec Corporation
#Sales Analysis ETL, etc,......

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from fpdf import FPDF
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict, Annotated
import requests


class State(TypedDict):
    data: Annotated[dict, lambda old, new: new]
    metrics: Annotated[dict, lambda old, new: new]
    rules: Annotated[dict, lambda old, new: new]

workflow = StateGraph(State)

# --- my Configuration ---
EXCEL_PATH = r"E:\Asiatec\Online_retail.xlsx"
OUTPUT_PDF = "sales_analysis_report.pdf"

# Ollama Configuration
OLLAMA_API = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"  # You can use "mistral", "phi3", etc.

# We can give Prompt Template for Strategy Generation, dialogue etc....here!
STRATEGY_PROMPT = """
Based on the following sales analysis:

- Best Customers: {top_customers}
- Best Products: {top_products}
- CAC Increased: {cac_increased}
- Sales Growth: {sales_growth}

Provide strategic business recommendations to increase profitability and customer retention.
"""

# --- How to process data? ---
def load_and_preprocess_data(filepath):
    df = pd.read_excel(filepath)
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalSales'] = df['Quantity'] * df['UnitPrice']
    return df


# --- What KPI we require? What next? ---
def calculate_key_metrics(df):
    customer_freq = df.groupby('CustomerID').size().sort_values(ascending=False).head(10)
    product_sales = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
    avg_order_value = df.groupby('InvoiceNo')['TotalSales'].sum().mean()
    cac_increased = avg_order_value > 50

    return {
        'top_customers': customer_freq,
        'top_products': product_sales,
        'cac_increased': cac_increased,
        'sales_growth': True
    }


# --- Association Rule Mining (apriori)---
def perform_association_rule_mining(df):
    basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0).map(lambda x: 1 if x > 0 else 0)
    basket = basket.astype(bool)

    frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
    if frequent_itemsets.empty:
        print("No frequent itemsets found.")
        return pd.DataFrame()

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2).sort_values(by=["lift"], ascending=False)
    return rules.head(10)


# --- Ollama LLM Call Function ---
def ollama_generate(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "max_tokens": 300
        }
    }

    try:
        response = requests.post(OLLAMA_API, json=payload)
        if response.status_code == 200:
            return response.json()['response']
        else:
            print(f"Ollama Error {response.status_code}: {response.text}")
            return "Failed to generate AI recommendations due to Ollama API error."
    except Exception as e:
        print("Exception:", e)
        return "AI strategy generation failed. Please ensure Ollama is running locally."


# --- What kind od PDF report we want? Any elements? ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Sales Analysis Report', align='C')
        self.ln(10)

    def add_section(self, title, content):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title)
        self.ln(6)
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, content)
        self.ln(6)

    def add_table(self, headers, rows):
        col_width = (self.w - 20) / len(headers)  # Page width minus margins
        self.set_font('Arial', 'B', 10)
        for header in headers:
            self.cell(col_width, 7, header, border=1)
        self.ln()
        self.set_font('Arial', '', 10)
        for row in rows:
            for item in row:
                self.cell(col_width, 6, str(item), border=1)
            self.ln()


def generate_pdf_report(metrics, rules):
    pdf = PDF()
    pdf.add_page()

    # Page 1: Summary
    pdf.add_section("Summary", "This report provides insights into top-performing customers, products, and recommendations based on 2010 sales data.")

    # Page 2: Top Customers and Products
    pdf.add_page()
    pdf.add_section("Top 10 Customers", "\n".join([f"Customer ID {cid}: {count} orders" for cid, count in metrics['top_customers'].items()]))
    pdf.add_section("Top 10 Products", "\n".join([f"{prod}: {qty}" for prod, qty in metrics['top_products'].items()]))

    # Page 3: Strategy Recommendations
    pdf.add_page()
    strategy_prompt = STRATEGY_PROMPT.format(
        top_customers=", ".join(map(str, metrics['top_customers'].index.tolist())),
        top_products=", ".join(map(str, metrics['top_products'].index.tolist())),
        cac_increased=str(metrics['cac_increased']),
        sales_growth=str(metrics['sales_growth'])
    )
    strategy = ollama_generate(strategy_prompt)
    pdf.add_section("Strategic Recommendations", strategy)

    # Page 4: Association Rules
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Frequent Product Associations", ln=True, align="C")
    
    if not rules.empty:
        headers = ["Antecedents", "Consequents", "Support", "Confidence", "Lift"]
        rows = [[str(r.antecedents), str(r.consequents), round(r.support, 3), round(r.confidence, 3), round(r.lift, 3)] for _, r in rules.iterrows()]
        pdf.add_table(headers, rows)
    else:
        pdf.cell(0, 10, "No association rules found.", ln=True)

    pdf.output(OUTPUT_PDF)
    print(f"Report generated at: {OUTPUT_PDF}")


# --- what nodes we need? what else? ---
def input_node(state):
    state["data"] = load_and_preprocess_data(EXCEL_PATH)
    return state

def processing_node(state):
    state["metrics"] = calculate_key_metrics(state["data"])
    return state

def association_node(state):
    state["rules"] = perform_association_rule_mining(state["data"])
    return state

def recommendation_node(state):
    generate_pdf_report(state["metrics"], state["rules"])
    return state


# --- our Workflow ---
workflow.add_node("input_node", input_node)
workflow.add_node("processing_node", processing_node)
workflow.add_node("association_node", association_node)
workflow.add_node("recommendation_node", recommendation_node)

workflow.set_entry_point("input_node")
workflow.add_edge("input_node", "processing_node")
workflow.add_edge("processing_node", "association_node")
workflow.add_edge("association_node", "recommendation_node")
workflow.add_edge("recommendation_node", END)

app = workflow.compile()

# Run our agent
if __name__ == "__main__":
    initial_state = {"data": {}, "metrics": {}, "rules": {}}
    final_state = app.invoke(initial_state)