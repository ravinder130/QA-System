from rag_chatbot import answer_question, retrieve_documents


def retrieve(question: str, top_k: int = 5):
    """
    Test only the retrieval component of the RAG system.
    """

    if not isinstance(question, str):
        raise TypeError("Question must be a string.")

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    return retrieve_documents(question, top_k=top_k)


def ask_chatbot(question: str, top_k: int = 5) -> str:
    """
    Run the complete RAG pipeline:
    retrieval + answer generation.
    """

    return answer_question(question, top_k=top_k)["answer"]