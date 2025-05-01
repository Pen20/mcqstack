import os
import json
import pandas as pd
import traceback
import PyPDF2
from dotenv import load_dotenv
from src.mcqstack.utils import read_file, get_table_data
from src.mcqstack.logger import logging

# Importing necessary packages from langchain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# Load environment variables from the .env file
load_dotenv()

# Access the environment variables just like you would with os.environ
KEY = os.getenv("my_openai_key")

# Import OpenAI key to get the LLM
llm = ChatOpenAI(openai_api_key=KEY, model_name="gpt-3.5-turbo", temperature=0.5)

TEMPLATE = """
Text: {text}
You are an expert MCQ maker. Given the above text, it is your job \
to create a quiz of {number} multiple choice questions for {subject} students in a {tone} tone. \
Make sure the questions are not repeated and that all questions are consistent with the text. \
Use the RESPONSE_JSON below as a guide for formatting. Ensure you create exactly {number} MCQs.

### RESPONSE_JSON
{response_json}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=TEMPLATE
)

quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)

# Develop an evaluation template for my quizzes.
TEMPLATE2 = """
You are an expert English grammarian and writer. Given a Multiple Choice Quiz for {subject} students, \
you need to evaluate the complexity of the questions and provide a concise analysis (max 50 words) of the quiz's overall difficulty. \
If any questions do not align with the cognitive and analytical abilities of the students, \
update those questions and adjust the tone so that the quiz perfectly fits their abilities.

Quiz_MCQs:
{quiz}

Provide your expert review below:
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],
    template=TEMPLATE2
)

review_chain = LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)

# This is an Overall Chain where we run the two chains in Sequence
generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain, review_chain],
    input_variables=["text", "number", "subject", "tone", "response_json"],
    output_variables=["quiz", "review"],
    verbose=True,
)

# Optional: wrap in a function for modularity
def generate_quiz_and_review(text, number, subject, tone, response_json):
    return generate_evaluate_chain({
        "text": text,
        "number": number,
        "subject": subject,
        "tone": tone,
        "response_json": json.dumps(response_json)
    })
