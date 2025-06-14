import streamlit as st
from qa_chain import ask_question

st.set_page_config(page_title="Codebase QA Chatbot", layout="centered")

st.title("💬 Codebase QA Chatbot")
st.write("Ask questions about the code repository!")

query = st.text_input("Enter your question:")
if st.button("Submit") or query:
    if query.strip():
        with st.spinner("Thinking..."):
            answer = ask_question(query)
        st.markdown("### 📌 Answer")
        st.write(answer)
    else:
        st.warning("Please enter a valid question.")
