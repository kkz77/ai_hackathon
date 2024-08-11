import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate 
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import find_dotenv, load_dotenv
import base64

# Load environment variables
load_dotenv(find_dotenv())
google_api_key = os.getenv("GOOGLE_API_KEY")
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''

#Define current directory
current_dir = os.path.dirname(os.path.abspath(__file__))


# Initialize the LLM and Gemini_embeddings with the Google API key
llm = GoogleGenerativeAI(model="models/gemini-pro", google_api_key=google_api_key)

# Define the prompt template
template ="""
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
prompt = PromptTemplate(template=template, input_variables=["history", "input"])

# Initialize memory for the conversation
memory = ConversationBufferMemory()

# Initialize the conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
)

def chatbot(human_input, image = None):
    if image:
        # Encode the image to base64
        with open(image, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        # Prepare a prompt that includes the image
        prompt = f"Here's an image for analysis: [IMAGE]{image}[/IMAGE]\n\nHuman input: {human_input}"
        
    else:
        prompt = human_input
    # Generate a response from the LLM
    response = conversation.predict(input=prompt)
    return response
