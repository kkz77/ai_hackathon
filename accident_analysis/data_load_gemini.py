import faiss
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import base64
import numpy as np
import os
import pickle

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Retrieve Gemini API credentials from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")

# Define current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize Gemini LLM and embeddings
llm = GoogleGenerativeAI(model="models/gemini-1.5-pro", google_api_key=google_api_key)
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=google_api_key)

# Load the FAISS index and metadata
index_path = os.path.join(current_dir, 'faiss_index.index')
index = faiss.read_index(index_path)

metadata_file = os.path.join(current_dir, 'faiss_metadata.pkl')
with open(metadata_file, 'rb') as f:
    metadata = pickle.load(f)

# Check embedding dimensionality
sample_text = "This is a sample text to check embedding dimensionality."
sample_embedding = np.array(gemini_embeddings.embed_query(sample_text))
print("Sample embedding shape:", sample_embedding.shape)

# Ensure the embedding dimensionality matches the FAISS index
if sample_embedding.shape[0] != index.d:
    print("Embedding dimensionality does not match the FAISS index. Recreating the FAISS index.")
    
    # Load PDF documents from directory
    loader = PyPDFDirectoryLoader("./accident_analysis/Data/")
    documents = loader.load()

    # Split documents into smaller chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
    )
    docs = text_splitter.split_documents(documents)

    # Create a new FAISS vector store from the document embeddings
    vectorstore_faiss = FAISS.from_documents(docs, gemini_embeddings)

    # Save the new FAISS index and metadata
    faiss.write_index(vectorstore_faiss.index, index_path)

    metadata = {
        'docstore': vectorstore_faiss.docstore,
        'index_to_docstore_id': vectorstore_faiss.index_to_docstore_id
    }
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
else:
    # Create the FAISS vector store using the loaded index and metadata
    vectorstore_faiss = FAISS(
        index=index,
        docstore=metadata['docstore'],
        index_to_docstore_id=metadata['index_to_docstore_id'],
        embedding_function=gemini_embeddings
    )

# Wrap the vector store for use with LangChain
wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)

# Load all PDF files from the specified directory
accident_file = os.path.join(current_dir, 'Data')
print("Accident file:", accident_file)
loader = PyPDFDirectoryLoader(accident_file)
accident_cases = loader.load()

# Define prompt templates for accident analysis and safety recommendations
prompt_template = """
Human: You are an industrial safety expert. You will do an accident case analysis of the accident case provided by the user as "Question". Use the following pieces of context to get additional knowledge. Based on the knowledge from the context and your pretrained general knowledge, you have to answer three questions about the accident:
1. First summarize the provided case and describe it.
2. Potential accident level with percentage.
3. Why this accident occurred based on this theory:
    (Petersen’s Theory states that the causes of accidents-incidents are human error and/or system failure. Overload, Traps, and decision to error lead to human error. Human error can directly cause accidents or contribute to system failure. System failure can in turn cause accidents/incidents. Traps can arise from defective workstation, improper design, and incompatible displays or controls.)
4. How to prevent recurrences:
    You must think about prevention methods according to:
    (1) Engineering control
    (2) Administrative control
    (3) PPE
<context>
{context}
</context>

Question: {question}

Assistant:
"""

prompt_template2 = """
Human: You are an industrial safety expert. Use the following pieces of context 
to provide appropriate health and safety recommendations to the worker. You should use
both your pretrained general knowledge and the knowledge from the context you read. 
Don't say "Based on the context". Show step by step if it is necessary. Make sure you use simple language the worker can understand.
<context>
{context}
</context>

Question: {question}

Assistant:
"""

conversation_template ="""
H: You are industrial safety expert. Imagine you are having a conversation with industrial worker. Provide appropriate health and safety recommendation
related to industrial fields based on your knowledge. When the user asks for health and safety recommendations,
you need to think step by step and analyze the case and generate precise content.
The AI provides specific details when necessary, but keeps responses concise when possible.
Don't say "Based on the context" or "Based on your knowledge". Don't let the answer be too long. 
If you need any answer for details analysis, ask the user back what you need.
The AI can analyze images when they are provided and detect machines.
Make sure you use simple language so that user can understand.
Current conversation:
<conversation_history>
{history}
</conversation_history>

Here is the human's next input, which may include an image:
<human_input>
{input}
</human_input>

Please provide a concise response unless a longer explanation is necessary. If an image is present, analyze it in the context of the human's input.

A:
"""


# Create the prompt
conversation_prompt = PromptTemplate(template=conversation_template, input_variables=["history", "input"])

# Initialize memory for the conversation
memory = ConversationBufferMemory()

# Initialize the conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=conversation_prompt,
)

# Initialize prompt templates
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
PROMPT2 = PromptTemplate(template=prompt_template2, input_variables=["context", "question"])

# Initialize RetrievalQA chains with the prompts
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

qa2 = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT2}
)

# Function to get context from RAG
def get_retrieved_context(user_input):
    return qa({"query": user_input})['result']

# Function to get accident analysis based on user input
def rgl(user_input):
    query = user_input 
    query_embedding = vectorstore_faiss.embedding_function.embed_query(query)
    print(np.array(query_embedding))
    return qa({"query": query})['result']

# Function to get safety recommendations based on user input and specific machine
def rgl2(user_input, machine=None):
    query = "This is " + machine + " " + user_input 
    query_embedding = vectorstore_faiss.embedding_function.embed_query(query)
    print(np.array(query_embedding))
    return qa2({"query": query})['result']

# Function to handle chatbot responses with RAG
def chatbot(human_input, image=None):
    context = get_retrieved_context(human_input)
    
    if image:
        # Encode the image to base64
        with open(image, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        # Prepare a prompt that includes the image
        prompt = f"Here's an image for analysis: [IMAGE]{encoded_image}[/IMAGE]\n\nHuman input: {human_input}\n\nContext: {context}"
    else:
        prompt = f"Human input: {human_input}\n\nContext: {context}"

    # Generate a response from the LLM
    response = conversation.predict(input=prompt)
    return response


# Example usage
#print(rgl("What can happen if I use a drilling machine"))
