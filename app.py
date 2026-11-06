import streamlit as st

from rag_chatbot import LocalRAG, answer_question

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📚", layout="wide")
st.title("📚 PDF RAG Chatbot")
st.caption("Ask questions about the PDFs located in this folder.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input(
    "Ask a question about the PDFs",
    placeholder="Example: What are the main AI job trends described in these reports?",
)

if st.button("Ask") and question.strip():
    with st.spinner("Searching the PDFs and retrieving the most relevant passages..."):
        result = answer_question(question, top_k=5)

    st.session_state.chat_history.append({"question": question, "answer": result["answer"]})

    st.markdown("### Answer")
    st.write(result["answer"])

    if result.get("sources"):
        st.markdown("### Relevant passages")
        for index, source in enumerate(result["sources"], start=1):
            with st.expander(f"{index}. {source['pdf']}"):
                st.write(source["text"])

if st.session_state.chat_history:
    st.markdown("### Recent questions")
    for item in reversed(st.session_state.chat_history[-5:]):
        st.markdown(f"**Q:** {item['question']}")
        st.markdown(f"**A:** {item['answer']}")
        st.write("---")
