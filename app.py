# if using OpenAI, uncomment the next line
# from langchain_openai import OpenAIEmbeddings
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.documents import Document
# if using OpenAI, uncomment the next line
# from langchain_community.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.runnables import RunnableLambda, RunnablePassthrough


from htmlTemplates import bot_template, user_template, css




def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 800,
        chunk_overlap = 100,
        length_function = len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(text_chunks):
    # If using OpenAI embeddings, uncomment the next line
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name = "hkunlp/instructor-xl") 
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding = embeddings)
    return vectorstore

# Old version using ConversationalRetrievalChain
#def get_conversation_chain(vector_store):
#    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
#    memory = ConversationBufferMemory(
#        memory_key = "chat_history", return_messages = True)
#    conversation_chain = ConversationalRetrievalChain.from_llm(
#        llm = llm,
#        retriever = vector_store.as_retriever(),
#        memory = memory
#    )
#       return conversation_chain

def get_conversation_chain(vector_store):
    llm = Ollama(
        model="llama3",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful assistant. Use the following context to answer the question.\n\n{context}"
        ),
        ("human", "{input}")
    ])

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    # --- RAG chain ---
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": RunnableLambda(
                lambda x: format_docs(retriever.invoke(x["input"]))
            ),
            "input": RunnableLambda(lambda x: x["input"]),
        }
        | prompt
        | llm
    )

    # --- Chat history store ---
    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

# Old version using ConversationalRetrievalChain
#def handle_userinput(user_question):
#    # Upload check
#    if st.session_state.conversation is None:
#        st.warning("Please upload and process your PDFs first before asking a question.")
#        return
#
#    response = st.session_state.conversation.invoke(
#        {"input": user_question},
#        config={"configurable": {"session_id": "default"}}
#    )
#    st.session_state.chat_history = response['chat_history']
#    for i, message in enumerate(st.session_state.chat_history):
#        if i % 2 == 0:
#            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
#        else:
#            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process your PDFs first before asking a question.")
        return

    # Ensure messages list exists
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # User bubble
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    # Get response from chain
    result = st.session_state.conversation.invoke(
        {"input": user_question},
        config={"configurable": {"session_id": "default"}}
    )

    # Ollama returns a string
    answer = result if isinstance(result, str) else str(result)

    # Bot bubble
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # Render chat
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.write(
                user_template.replace("{{MSG}}", msg["content"]),
                unsafe_allow_html=True
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", msg["content"]),
                unsafe_allow_html=True
            )






def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "messages" not in st.session_state or st.session_state.messages is None:
        st.session_state.messages = []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your upload:")
    if user_question:
        handle_userinput(user_question)

    # New process check
    if "processed" not in st.session_state:
        st.session_state.processed = False

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here! Then click on 'Process'", type=["pdf"], accept_multiple_files = True)
        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file before processing!")
                return

            with st.spinner("Processing"):
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)
                # Test point: Uncomment to see raw text
                # st.write(raw_text)

                # Get text chunks
                text_chunks = get_text_chunks(raw_text)
                # Test point: Uncomment to see text chunks
                # st.write(text_chunks)
                if not text_chunks:
                    st.error("Could not extract any text from the uploaded PDFs. Please check the files.")
                    return

                # Create vector store
                vector_store = get_vector_store(text_chunks)

                # Creat conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)

                st.session_state.processed = True
        if st.session_state.processed:
            st.success("✅ Process complete! You can now ask questions about your documents.")



if __name__ == '__main__':
    main()