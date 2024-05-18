import os
from azure.storage.blob import BlobServiceClient
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores.azuresearch import AzureSearch
import streamlit as st
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from tqdm import tqdm
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import SearchIndex, SearchField, SimpleField, SearchableField
from langchain.chat_models import AzureChatOpenAI  # Ensure this import is present

# Define Azure Storage connection details
connection_string = "DefaultEndpointsProtocol=https;AccountName=ajtsdevsistrg;AccountKey=oaY0INq+3IRBDzuD5tO1Av9YVloi8fs3CHecExkrPZ4r6PRRaThCKL3BZ2vL4V5+w+bqp/Ai8NVT+AStTBOY3g==;EndpointSuffix=core.windows.net"
container_name = "genai"

# Initialize Azure Blob Storage connection
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

# Initialize Azure Search vector store details
vector_store_address = "https://ai-search-ajuserv.search.windows.net"
vector_store_password = "9ODMu6tpu6y3B6nYzVd9bTabY8qVLL0FWnWRTxpJwrAzSeCFUbYa"
index_name = "langchain-vector-demo"

# Option 2: Use AzureOpenAIEmbeddings with an Azure account
azure_endpoint = "https://ajuserv-open-ai.openai.azure.com/"
azure_openai_api_key = "e6c4aed862ff419a8708632cb8837760"
azure_openai_api_version = "2023-12-01-preview"
azure_deployment_qa = "gpt-35-turbo-instruct-dev"

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_version=azure_openai_api_version,
    azure_endpoint=azure_endpoint,
    api_key=azure_openai_api_key,
)

# Create vector store instance
vector_store = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)

text_splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=20)

# Configure OpenAI settings
os.environ["OPENAI_API_KEY"] = "e6c4aed862ff419a8708632cb8837760"
os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ajuserv-open-ai.openai.azure.com/"

llm_azure = AzureChatOpenAI(
    deployment_name="gpt-35-turbo",
    temperature=0
)

qa = RetrievalQA.from_chain_type(llm=llm_azure, chain_type="stuff", retriever=vector_store.as_retriever())

# Function to transcribe audio files from Azure Blob Storage and get bot response
def transcribe_and_get_bot_response():
    blobs = container_client.list_blobs()
    num_files = sum(1 for blob in blobs if blob.name.endswith(('.m4a', '.mpeg', '.mp3')))
    
    with tqdm(total=num_files, desc="Transcribing Files") as pbar:
        for blob in blobs:
            if blob.name.endswith(('.m4a', '.mpeg', '.mp3')):
                filename = os.path.basename(blob.name)
                
                try:
                    os.makedirs("temp", exist_ok=True)
                    download_path = f"temp/{filename}"
                    with open(download_path, "wb") as audio_file:
                        blob_client = container_client.get_blob_client(blob.name)
                        audio_data = blob_client.download_blob().readall()
                        audio_file.write(audio_data)
                    
                    transcription = f"Transcription for {blob.name}"
                    bot_response = qa.run(query=transcription)
                    
                    st.write(f"Transcription for {blob.name}: {transcription}")
                    st.write("Chatterbean:", bot_response)
                    
                    pbar.update(1)
                except Exception as e:
                    st.error(f"An exception occurred at file {blob.name}: {str(e)}")

# Main function for Streamlit app
def main():
    st.title("CabCritique: Shaping Service Excellence")
    
    if st.button("Transcribe and Get Bot Response"):
        transcribe_and_get_bot_response()
    
    user_query = st.text_input("Privileged to assist with your queries")
    
    if st.button("Ask"):
        bot_response = qa.run(query=user_query)
        st.write("Chatterbean:", bot_response)

if __name__ == "__main__":
    main()
