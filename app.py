import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import faiss
import os
from langchain.vectorstores import VectorStore
import pickle
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI,OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "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"
st.set_page_config(page_title="Educhat")
st.header("Educhat")
subject = st.selectbox('Pick your subject', ['Chemistry_v1', 'Chemistry_v2','Physics_v1','Physics_v2','Maths_v1','Maths_v2','Computer','English'])
input=st.text_input("Input Prompt: ",key="input")
submit=st.button("submit")
print(subject)
embedding = OpenAIEmbeddings(openai_api_key = "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")
def choose_subject(subject):
    if subject == "Chemistry_v1":
        vectorstore_file = "Embedding_files/chemistry_v1/"
        local_vectorstore = FAISS.load_local(vectorstore_file, embedding,allow_dangerous_deserialization=True)
        return local_vectorstore
    elif subject == "Chemistry_v2":
        vectorstore_file = "Embedding_files/chemistry_v2/"
        local_vectorstore = FAISS.load_local(vectorstore_file, embedding,allow_dangerous_deserialization=True)
        return local_vectorstore
    elif subject == "Physics_v1":
        vectorstore_file = "Embedding_files/physics_v1/"
        local_vectorstore = FAISS.load_local(vectorstore_file, embedding,allow_dangerous_deserialization=True)
        return local_vectorstore
    elif subject == "Physics_v2":
        vectorstore_file = "Embedding_files/physics_v2"
        local_vectorstore = FAISS.load_local(vectorstore_file, embedding,allow_dangerous_deserialization=True)
        return local_vectorstore
    elif subject == "Maths_v1":
        vectorstore_file = "Embedding_files/maths_v1"
        local_vectorstore = FAISS.load_local(vectorstore_file, embedding,allow_dangerous_deserialization=True)
        return local_vectorstore
    elif subject == "Maths_v2":
        vectorstore_file = "Embedding_files/maths_v2"
        local_vectorstore = FAISS.load_local(vectorstore_file, embedding,allow_dangerous_deserialization=True)
        return local_vectorstore
    elif subject == "Computer":
        vectorstore_file = "Embedding_files/computer"
        local_vectorstore = FAISS.load_local(vectorstore_file, embedding,allow_dangerous_deserialization=True)
        return local_vectorstore
    elif subject == "English":
        vectorstore_file = "Embedding_files/english"
        local_vectorstore = FAISS.load_local(vectorstore_file, embedding,allow_dangerous_deserialization=True)
        return local_vectorstore

prompt = """
Your task is to assist Class 12 students in understanding concepts and solving their questions across the subjects of Chemistry, Physics, Maths, English, and Computer Science. Use the information from textbooks stored in the vector database to provide an accurate and educational response.
Clarity and Conciseness: Craft responses that are clear and to the point, ideally within 100 words, focusing on the core concept or answer. Avoid lengthy explanations unless essential.
Use of Analogies: Where applicable, provide relatable analogies or comparisons to everyday situations. These should make complex ideas simpler and easier to visualize. For example, if explaining chemical bonding, you might compare it to people forming friendships based on shared interests.
Accuracy and Consistency: Ensure all responses align with the Class 12 textbook curriculum, adhering to factual accuracy and the standard scientific or literary principles relevant to the question.
Interactive Tone: Address students respectfully and engagingly, as if explaining directly to a peer. Avoid overly formal or technical jargon unless the question specifically requires it. Define any specialized terms briefly and simply.
Step-by-Step Guidance: If a question involves problem-solving, provide step-by-step guidance, but keep it as concise as possible. For mathematical or scientific calculations, show key steps without excessive detail.
Subject-Specific Focus: Tailor the response style to each subject. For example, in English, you might focus on themes or literary devices; in Physics or Maths, prioritize formulas and principles; in Chemistry, focus on reactions and atomic interactions; and in Computer Science, explain code snippets or programming concepts in an accessible way.
Respond in a way that fosters understanding and confidence in the student, ensuring they grasp both the answer and the underlying principle
"""




if submit:

    index = choose_subject(subject)
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
    memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever= index.as_retriever(),
            memory=memory
        )
    query = input
    result = conversation_chain({"question": prompt + query})
    answer = result["answer"]
    st.write(answer)
    #print(input,subject)