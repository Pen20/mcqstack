from setuptools import find_packages, setup

setup(
    name='mcqstack',
    version='0.0.1',
    author='motognon wastalas dogbalou',
    author_email='wastalasdassise@gmail.com',
    install_requires=["openai","langchain","streamlit","python-dotenv","langchain_community","PyPDF2"],
    packages=find_packages()
)