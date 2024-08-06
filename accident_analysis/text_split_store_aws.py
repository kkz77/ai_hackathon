import os
import re
import sys
import warnings
import boto3
import botocore
import numpy as np
import textwrap
from io import StringIO
from typing import Optional
from dotenv import load_dotenv
from langchain.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader, PyPDFDirectoryLoader
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
import faiss
import pickle

# Load environment variables from .env file
load_dotenv()

# Retrieve AWS credentials from environment variables
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Initialize Boto3 client for Bedrock
boto3_bedrock = boto3.client('bedrock-runtime',
                             aws_access_key_id=AWS_ACCESS_KEY_ID,
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                             region_name=AWS_DEFAULT_REGION)

# Suppress warnings
warnings.filterwarnings('ignore')

# Function to print text with word wrap
def print_ww(*args, width: int = 100, **kwargs):
    """Like print(), but wraps output to `width` characters (default 100)"""
    buffer = StringIO()
    try:
        _stdout = sys.stdout
        sys.stdout = buffer
        print(*args, **kwargs)
        output = buffer.getvalue()
    finally:
        sys.stdout = _stdout
    for line in output.splitlines():
        print("\n".join(textwrap.wrap(line, width=width)))

# Configure the language model and embeddings
llm = BedrockChat(model_id="anthropic.claude-3-sonnet-20240229-v1:0", client=boto3_bedrock, model_kwargs={'max_tokens_to_sample': 200})
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=boto3_bedrock)

# Load PDF documents from directory
loader = PyPDFDirectoryLoader("./accident_analysis/Data/")
documents = loader.load()

# Split documents into smaller chunks for processing
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000,
    chunk_overlap=1000,
)
docs = text_splitter.split_documents(documents)

# Calculate average document length before and after splitting
avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents]) // len(documents)
avg_char_count_pre = avg_doc_length(documents)
avg_char_count_post = avg_doc_length(docs)

print(f'Average length among {len(documents)} documents loaded is {avg_char_count_pre} characters.')
print(f'After the split we have {len(docs)} documents more than the original {len(documents)}.')
print(f'Average length among {len(docs)} documents (after split) is {avg_char_count_post} characters.')

# Generate a sample embedding to check the setup
try:
    sample_embedding = np.array(bedrock_embeddings.embed_query(docs[0].page_content))
    print("Sample embedding of a document chunk: ", sample_embedding)
    print("Size of the embedding: ", sample_embedding.shape)
except ValueError as error:
    if "AccessDeniedException" in str(error):
        print(f"\x1b[41m{error}\nTo troubleshoot this issue please refer to the following resources.\n"
              "https://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\n"
              "https://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
        class StopExecution(ValueError):
            def _render_traceback_(self):
                pass
        raise StopExecution        
    else:
        raise error

# Create a FAISS vector store from the document embeddings
vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)

# Save the FAISS index and metadata
faiss_index_file = 'faiss_index.index'
faiss.write_index(vectorstore_faiss.index, faiss_index_file)

metadata_file = 'faiss_metadata.pkl'
metadata = {
    'docstore': vectorstore_faiss.docstore,
    'index_to_docstore_id': vectorstore_faiss.index_to_docstore_id
}
with open(metadata_file, 'wb') as f:
    pickle.dump(metadata, f)

print("FAISS index and metadata saved locally.")

# Wrap the vector store for use with LangChain
wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)
