"""
This is a PoC to demo how one can upload a csv, in my case a network web server logs csv and chat with it for answers.
Remeber: LLM answer will be only as good as your query. Please be clear and to the point when asking questions.
"""

import streamlit as st
import pandas as pd
import os
import json
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

def display_json_as_table(json_data):
    """Convert JSON to a DataFrame and render as HTML table."""
    df = pd.DataFrame(json.loads(json_data))
    html_table = df.to_html(index=False, escape=False, classes="table table-bordered table-condensed")
    return html_table

# Streamlit App
st.set_page_config(page_title="WebLogParserBot")
st.title("CSV Analysis for web access logs with LLM's and Pandas Visualizations")
st.write("Dataset: https://www.kaggle.com/datasets/kzmontage/e-commerce-website-logs")

# Sidebar for inputs
st.sidebar.header("Settings")
openai_api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Placeholder for the main DataFrame
df = None
# Process uploaded CSV
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Successfully loaded '{uploaded_file.name}'")
        st.sidebar.write("**Preview of Uploaded CSV (Top 3 Rows):**")
        st.sidebar.dataframe(df.head(3))
    except Exception as e:
        st.sidebar.error(f"Error loading CSV: {e}")

# Generate LangChain agent if API key and file are available
if openai_api_key and df is not None:
    try:
        # Save uploaded file temporarily
        temp_csv_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        df.to_csv(temp_csv_path, index=False)

        # Query input on the main page at the bottom
        query = st.text_input("Enter your query for the CSV data:")
        if query:
            with st.spinner("Processing your query..."):
                agent = create_pandas_dataframe_agent(
                    ChatOpenAI(temperature=0, api_key=openai_api_key),
                    df,
                    verbose=True,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    allow_dangerous_code=True
                )
                
                

                # Append JSON format instruction
                response = agent.invoke(f"""Output ONLY in JSON format. 
                                        Instruction: Use only the retrieval result for your response. User Query: " +{query}  + "JSON output:""")
                json_data = response["output"].replace("```json", "").replace("```", "").strip()
                if isinstance(json_data, str):
                    json_data = "[" + json_data + "]"  # Ensure JSON is a list format

                # Convert JSON to table and display
                html_table = display_json_as_table(json_data)
                st.markdown(
                    f"""
                    <style>
                    .table {{
                        width: 50% !important;
                        margin: 0 auto !important;
                    }}
                    .table-bordered {{
                        border-collapse: collapse !important;
                    }}
                    .table-condensed th, .table-condensed td {{
                        padding: 5px !important;
                    }}
                    </style>
                    """, unsafe_allow_html=True
                )
                st.write(html_table, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error initializing LangChain agent: {e}")

# Visualization section on the main panel on complete csv
if df is not None:
    st.write("### Regular Pandas Visualization into Bar / Line / Pie Charts")
    columns = df.columns.tolist()

    # Dropdowns for visualization
    col_for_vis = st.selectbox("Select a column to visualize:", ["---select---"] + columns)
    vis_type = st.selectbox("Select Visualization Type:", ["Bar Chart", "Pie Chart", "Line Chart"])

    # Generate visualization
    if col_for_vis != "---select---":
        st.write(f"### {vis_type} for '{col_for_vis}'")
        try:
            if vis_type == "Bar Chart":
                st.bar_chart(df[col_for_vis].value_counts())
            elif vis_type == "Pie Chart":
                st.write(df[col_for_vis].value_counts().plot.pie(autopct="%1.1f%%"))
                st.pyplot()
            elif vis_type == "Line Chart":
                st.line_chart(df[col_for_vis])
        except Exception as e:
            st.error(f"Error generating {vis_type}: {e}")

"""
requirements.txt

streamlit
pandas
langchain
langchain_experimental
langchain-openai
"""

