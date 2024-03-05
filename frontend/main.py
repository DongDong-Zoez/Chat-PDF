import streamlit as st 
from typing import List, Dict
import requests
import json

def handle_userinput(user_question: str) -> None:
    """
    Handles user input, sends it to a server for processing,
    and updates the Streamlit app with the response.

    Parameters:
        user_question (str): The question asked by the user.

    Returns:
        None
    """
    with st.chat_message("user"):
        st.markdown(user_question)

    headers = {'Content-type': 'application/json'}
    response = requests.post(
        st.session_state.post_url,
        json.dumps({"question": user_question}),
        headers=headers
    )
    response = response.json()

    with st.chat_message("assistant"):
        st.markdown(response["answer"])

    chat_historys: List[Dict[str, str]] = []
    for item in response["chat_history"]:
        temp_dict: Dict[str, str] = {}
        for sub_item in item:
            temp_dict[sub_item[0]] = sub_item[1]
        temp_dict["role"] = "user" if temp_dict["type"] == "human" else "assistant"
        chat_historys.append(temp_dict)
    st.session_state.chat_history = chat_historys

def clear():
    requests.post(
        "http://wsgiserver:8000/api/v1/conversational_rag/clear_memory",
        {}
    )
    st.session_state.chat_history = []


def main():

    st.set_page_config(
        page_title="Chat with multiple PDFs",
        page_icon=":books:"
    )
    llm_url = "http://wsgiserver:8000/api/v1/conversational_rag/generate"
    st.session_state.post_url = llm_url
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.header("Chat with multiple PDFs")


    
    with st.chat_message("assistant"):
        st.markdown("Hi there!!")

        
    for history in st.session_state.chat_history:
        with st.chat_message(history["role"]):
            st.markdown(history["content"])

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", 
            accept_multiple_files=True
        )
        

        st.subheader("Generating parameters")
        temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
        top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.75, step=0.01)
        top_k = st.sidebar.slider('top_k', min_value=20, max_value=100, value=40, step=1)
        max_length = st.sidebar.slider('max_new_tokens', min_value=32, max_value=1280, value=1000, step=8)
        num_beams = st.sidebar.slider('num_beams', min_value=1, max_value=8, value=4, step=1)

        if st.button("Process"):
            with st.spinner("Initialize Chain ..."):
                response = requests.post(
                    "http://wsgiserver:8000/api/v1/conversational_rag/build_chain",
                    data = {
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "max_new_tokens": max_length,
                        "num_beams": num_beams,
                    },
                    files=[(f"file{i}", doc.getvalue()) for i, doc in enumerate(docs)],
                )

        st.button("Clear chat history", on_click=clear)

    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

if __name__ == "__main__":
    main()