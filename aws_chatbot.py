import boto3
import warnings
from io import StringIO
import sys
import textwrap
import os
import json
from typing import Optional
import base64
# External Dependencies:
from botocore.config import Config
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import BedrockChat
from langchain_core.messages import HumanMessage
from langchain.chains import ConversationChain
load_dotenv(override=True)

warnings.filterwarnings('ignore')
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
        

def get_bedrock_client(
    assumed_role: Optional[str] = None,
    region: Optional[str] = None,
    runtime: Optional[bool] = True,
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    assumed_role :
        Optional ARN of an AWS IAM role to assume for calling the Bedrock service. If not
        specified, the current active credentials will be used.
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-1").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    runtime :
        Optional choice of getting different client to perform operations with the Amazon Bedrock service.
    """
    if region is None:
        target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    else:
        target_region = region

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        print(f"  Using role: {assumed_role}", end='')
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=str(assumed_role),
            RoleSessionName="langchain-llm-1"
        )
        print(" ... successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        #client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    if runtime:
        service_name='bedrock-runtime'
    else:
        service_name='bedrock'

    bedrock_client = session.client(
        service_name=service_name,
        config=retry_config,
        **client_kwargs
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client


boto3_bedrock = get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region='us-west-2' #os.environ.get("AWS_DEFAULT_REGION", None)
)

# turn verbose to true to see the full logs and documents
modelId = "anthropic.claude-3-sonnet-20240229-v1:0" #"anthropic.claude-v2"

cl_llm = BedrockChat(
    model_id=modelId,
    client=boto3_bedrock,
    model_kwargs={
        "temperature": 0.5,
        "max_tokens": 1000,
        "anthropic_version": "bedrock-2023-05-31",
    },
)

conversation= ConversationChain(
    llm=cl_llm, verbose=False, memory=ConversationBufferMemory() #memory_chain
)

#conversation.predict(input="Hi there!")

# # langchain prompts do not always work with all the models. This prompt is tuned for Claude
claude_prompt = PromptTemplate.from_template("""
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
""")

conversation.prompt = claude_prompt

def chatbot(human_input, image=None):
    if image:
        # Encode the image to base64
        with open(image, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        # Prepare a prompt that includes the image
        prompt = f"Here's an image for analysis: [IMAGE]{image}[/IMAGE]\n\nHuman input: {human_input}"
    else:
        prompt = human_input
    ai_output = conversation.predict(input=prompt)
    return ai_output


# def chatbot(human_input):
#     while True:
#         #human_input = input("You: ")  # Get input from the console
#         if human_input.lower() == 'exit':
#             break
#         ai_output = conversation.predict(input=human_input)
#     return ai_output
# print("AI:",chatbot("hi"))
#print_ww(conversation.predict(input="Give me a few tips on how to start a new garden."))

# print_ww(conversation.predict(input="Cool. Will that work with tomatoes?"))