import os
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
from src.mcqstack.utils import read_file, get_table_data
import streamlit as st
from langchain.callbacks import get_openai_callback
from src.mcqstack.MCQMaths import generate_evaluate_chain
from src.mcqstack.logger import logging


# Load JSON file safely with correct path format
try:
    with open(r'C:\Users\Hp\Documents\github\mcqstack\Response.json', 'r') as file:
        RESPONSE_JSON = json.load(file)
except FileNotFoundError:
    st.error("Response.json file not found. Please check the file path.")
    st.stop()


# App title
st.title("AutoMath MCQ Maker with LangChain")


# Function to handle the MCQ generation process
def process_uploaded_file(uploaded_file, mcq_count, subject, tone, response_json):
    text = read_file(uploaded_file)
    with get_openai_callback() as cb:
        response = generate_evaluate_chain({
            "text": text,
            "number": mcq_count,
            "subject": subject,
            "tone": tone,
            "response_json": json.dumps(response_json)
        })
    return response, cb


# Form for user inputs
with st.form("user_inputs"):

    # File Upload
    uploaded_file = st.file_uploader("Upload a PDF or text file")

    # Input Fields
    mcq_count = st.number_input("No. of MCQs", min_value=3, max_value=50)

    # Subject
    subject = st.text_input("Insert Subject", max_chars=20)

    # Quiz Tone
    tone = st.text_input("Complexity Level of Questions", max_chars=30, value="Simple")

    # Add Button
    button = st.form_submit_button("Create MCQsMaTH")

    if button:
        if uploaded_file is None:
            st.error("Please upload a file before submitting.")
        elif not mcq_count or not subject or not tone:
            st.error("Please complete all the fields.")
        elif uploaded_file.type not in ["application/pdf", "text/plain"]:
            st.error("Please upload a valid PDF or text file.")
        else:
            with st.spinner("Generating MCQs..."):
                try:
                    response, cb = process_uploaded_file(
                        uploaded_file, mcq_count, subject, tone, RESPONSE_JSON
                    )

                except Exception as e:
                    traceback.print_exception(type(e), e, e.__traceback__)
                    st.error("An error occurred during processing.")
                else:
                    # Show token usage & cost
                    st.info(
                        f"""
                        **Token Usage:**
                        - Total tokens: {cb.total_tokens}
                        - Prompt tokens: {cb.prompt_tokens}
                        - Completion tokens: {cb.completion_tokens}
                        - Total cost: ${cb.total_cost:.4f}
                        """
                    )

                    # Process response
                    if isinstance(response, dict):
                        quiz = response.get("quiz", None)
                        if quiz is not None:
                            table_data = get_table_data(quiz)
                            if table_data is not None:
                                df = pd.DataFrame(table_data)
                                df.index = df.index + 1
                                st.table(df)
                                # Display review if available
                                st.text_area(label="Review", value=response.get("review", "No review available."))
                            else:
                                st.error("Failed to extract table data from the response.")
                        else:
                            st.warning("No quiz data found in the response.")
                            st.json(response)
                    else:
                        st.warning("Unexpected response format:")
                        st.json(response)
