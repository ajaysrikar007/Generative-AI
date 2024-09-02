import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
from dotenv import load_dotenv


load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
groq_api_key=os.getenv('GROQ_API_KEY')

st.title("Chatgroq With Llama3 Demo")

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)
def vector_embeddings():
    embeddings = OpenAIEmbeddings()
    loader = PyPDFDirectoryLoader("./us_census/")
    try:
        docs = loader.load()
    except Exception as e:
        st.error(f"Failed to load documents: {e}")
        return None, None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:20])
    
    print("Documents loaded: ", len(docs))
    print("Final documents: ", len(final_documents))
    
    if not final_documents:
        st.error("No final documents to process.")
        return None, None

    # Generate embeddings for the final documents
    # try:
    #     document_texts = [doc.page_content for doc in final_documents]
    #     document_embeddings = embeddings.embed_documents(document_texts)
    # except Exception as e:
    #     st.error(f"Failed to generate embeddings: {e}")
    #     return None, None
    
    # if not document_embeddings or len(document_embeddings) == 0:
    #     st.error("Failed to generate embeddings.")
    #     return None, None
    
    # if not isinstance(document_embeddings[0], list):
    #     st.error("Embeddings are not in the expected format.")
    #     return None, None
    
    # print("Embeddings generated: ", len(document_embeddings))
    
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors, final_documents

prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Documents Embeddirng"):
    vectors, final_documents = vector_embeddings()
    if vectors:
        st.write("Vector Store DB Is Ready")

if prompt1 and 'vectors' in locals() and vectors is not None:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    print("Response time:", time.process_time() - start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")

#vectors, final_documents = vector_embeddings()