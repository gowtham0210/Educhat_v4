from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

loader = PyPDFLoader("Files/physics_volume_2.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
text_splitter.split_documents(docs)[:5]

documents=text_splitter.split_documents(docs)
print(documents[10])
print(len(documents))

che_v2_db=FAISS.from_documents(documents,
                        OpenAIEmbeddings(openai_api_key = "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"))
che_v2_db.save_local("Embedding_files/physics_v2")



