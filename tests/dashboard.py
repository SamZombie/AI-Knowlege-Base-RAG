import streamlit as st
import json
import pandas as pd
from pathlib import Path

results_file = Path(__file__).resolve().parent / "eval_results.json"
with open(results_file, "r") as f:
    results = json.load(f)

avg_retrieval = results["avg_retrieval_score"]
avg_answer = results["avg_answer_score"]
question_summary = results["question_summary"]

st.title("RAG Evaluation Dashboard")

col1, col2 = st.columns(2)
with col1:
    st.metric("Average Retrieval Score", f"{avg_retrieval:.2f}")
with col2:
    st.metric("Average Answer Score", f"{avg_answer:.2f}")

st.divider()

st.subheader("Per Question Scores")
table_data = [
    {
        "Question": entry["question"],
        "Retrieval Score": entry["retrieval_score"],
        "Answer Score": entry["answer_score"]
    }
    for entry in question_summary
]
st.dataframe(pd.DataFrame(table_data), use_container_width=True)

st.divider()

st.subheader("Question Explorer")

selected_question = st.selectbox(
    "Select a question", options=[
        entry["question"] for entry in question_summary])

selected = next(
    entry for entry in question_summary if entry["question"] == selected_question)

st.text_area(
    "Expected Answer",
    value=selected["expected_answer"],
    disabled=True,
    height=150)
st.text_area(
    "Generated Answer",
    value=selected["generated_answer"],
    disabled=True,
    height=150)

col1, col2 = st.columns(2)
with col1:
    st.write("**Expected Sources**")
    for source in selected["expected_sources"]:
        st.write(f"- {source}")
with col2:
    st.write("**Retrieved Sources**")
    for source in selected["retrieved_sources"]:
        st.write(f"- {source}")
