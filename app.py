import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css,bot_template,user_template
from dotenv import load_dotenv

load_dotenv()
openai_api_key=os.getenv("OPEN_API_KEY")
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the text into chunks
def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks

# Function to get the vector
def get_vector(text_chunks,index_name="newindexx"):
    embedings = OpenAIEmbeddings(api_key=openai_api_key)
    vector_store = PineconeVectorStore.from_texts(
        text_chunks, embedings, index_name=index_name
    )
    return vector_store

def get_conservation_chain(vectorstore):
    model = ChatOpenAI(api_key=openai_api_key, temperature=0.2)
    memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    conversation_chain=ConversationalRetrievalChain.from_llm(llm=model,memory=memory,retriever=vectorstore.as_retriever())
    return conversation_chain


def handle_user_question(user_question):
    response=st.session_state.conversation({"question":user_question})
    st.session_state.chat_history=response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
        else:
             st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
    # st.write(response)

def main():
    load_dotenv()
    st.set_page_config("Chat with multiple pdfs")
    st.write(css,unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    st.header("Chat with multiple pdfs")
    user_question = st.text_input("Enter your question")
    if user_question:
        handle_user_question(user_question)
    st.write(user_template.replace("{{MSG}}"," Chatbot"),unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}"," User"),unsafe_allow_html=True)
    with st.sidebar:
        pdf_docs = st.file_uploader("Upload pdfs", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Process the pdfs
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                vector_store = get_vector(chunks)
                st.session_state.conversation=get_conservation_chain(vector_store)

if __name__ == "__main__":
    main()
