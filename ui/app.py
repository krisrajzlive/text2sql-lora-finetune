"""Streamlit front end for the text-to-SQL API."""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
EXAMPLE_SCHEMA = "CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary INT)"

st.title("Text-to-SQL")
schema = st.text_area("Table schema (CREATE TABLE ...)", EXAMPLE_SCHEMA, height=120)
question = st.text_input("Question", "How many employees are in the sales department?")

if st.button("Generate SQL", disabled=not (schema.strip() and question.strip())):
    try:
        r = requests.post(f"{API_URL}/generate", json={"question": question, "schema": schema}, timeout=300)
        r.raise_for_status()
        st.code(r.json()["sql"], language="sql")
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
