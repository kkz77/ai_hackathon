import gradio as gr
from dotenv import find_dotenv, load_dotenv
import os
from gtts import gTTS
import audio_helper
import tempfile
import aws_chatbot as ac
import gemini_chatbot as gc
import shutil
import PyPDF2
import io
import sys

# Set up directories and paths
current_dir = os.path.dirname(os.path.abspath(__file__))
accident_analysis_dir = os.path.join(current_dir, 'accident_analysis')
yolo_dir = os.path.join(current_dir,'fifth_yolo_python-1')
sys.path.append(accident_analysis_dir)
sys.path.append(yolo_dir)

# Load environment variables
load_dotenv(find_dotenv())

# Import local modules
import data_load_gemini as dl
##For aws service
#import data_load_aws as dl
import new_yolo as yolo

# Transcribe audio file to text
def transcribe_audio(audio_file):
    return audio_helper.transcribe_to_text(audio_file)

# Convert text to speech and save it as an audio file
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

# Save uploaded image to a specified directory
def save_uploaded_image(image):
    if image is not None:
        upload_dir = "uploaded_images"
        os.makedirs(upload_dir, exist_ok=True)
        filename = os.path.join(upload_dir, f"uploaded_image_{os.path.basename(image)}")
        shutil.copy(image, filename)
        print("file name = ", filename)
        return filename
    return None

# Play the latest response from the chat history
def play_latest_response(chat_history):
    if chat_history and chat_history[-1][0] == "Bot":
        latest_response = chat_history[-1][1]
        audio_file = text_to_speech(latest_response)
        return gr.Audio(value=audio_file, autoplay=True)
    return None

# Process audio input and update chat history
def process_audio(audio_file, chat_history):
    text = transcribe_audio(audio_file)
    return text, chat_history

# Process video, generate transcription, and perform analysis
def process_video_and_generate_text(video):
    if video is None:
        return "No video uploaded.", "" 
    transcribe_text, machine = yolo.yolo_media_upload(video)
    generated_text = "This machine is " + machine + ". " + dl.rgl2(transcribe_text, machine)
    return transcribe_text, generated_text

# Format chat history for display
def format_chat_history(chat_history):
    formatted_history = []
    for role, message in chat_history:
        if role == "User":
            formatted_history.append((None, message))
        else:
            formatted_history.append((message, None))
    return formatted_history

# Handle user input, save image if uploaded, and generate response using chatbot
def handle_input_modified(input_text, chat_history, image):
    chat_history.append(("User", input_text))
    image_path = save_uploaded_image(image) 
    if image_path:
        #For AWS chatbot claude model
        #response = ac.chatbot(input_text, image=image_path)
        response = gc.chatbot(input_text, image=image_path)
    else:
        #response = ac.chatbot(input_text)
        response = gc.chatbot(input_text)
    chat_history.append(("Bot", response))
    return format_chat_history(chat_history), chat_history, ""  # Only clear text input

# Process PDF file and extract text
def process_pdf(file):
    if file is None:
        return "No file uploaded."
    try:
        if hasattr(file, 'name'):
            with open(file.name, 'rb') as f:
                file_content = f.read()
        else:
            file_content = file

        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return dl.rgl(text)

    except Exception as e:
        return f"Error processing PDF: {str(e)}"

# Analyze accident case from either file or text input
def analyze_accident(file_or_text, is_file):
    if is_file:
        content = process_pdf(file_or_text)
    else:
        content = dl.rgl(file_or_text)
    return content

# Handle different input types for accident analysis
def handle_input(input_type, file, text):
    if input_type == "pdf":
        if file is not None:
            result = analyze_accident(file, is_file=True)
        else:
            result = "Please upload a PDF file."
    elif input_type == "text":
        if text.strip() != "":
            result = analyze_accident(text, is_file=False)
        else:
            result = "Please enter text to analyze."
    else:
        result = "Please select an input type."
    return result, gr.update(), ""  # Don't clear file input, only clear text input

# Define the Gradio interface
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("SpeechSync", id="tab-soundsync"):
            gr.Markdown("### SpeechSync")
            with gr.Row():
                video_input = gr.Video(label="Upload Video")
            
            generate_button = gr.Button("Generate Text")
            
            transcribe_output = gr.Textbox(label="Your Prompt", lines=3)
            text_output = gr.Textbox(label="AI Generated Text", lines=10)
            
            generate_button.click(
                fn=process_video_and_generate_text,
                inputs=video_input,
                outputs=[transcribe_output, text_output]
            )

        with gr.TabItem("Safety Bot", id="tab-safetybot"):
            gr.Markdown("### Safety Bot")
            with gr.Row():
                with gr.Column(scale=1):
                    chatbot = gr.Chatbot(layout="bubble", bubble_full_width=False, height=600)
                with gr.Column(scale=1):
                    text_input = gr.Textbox(label="Type your message here:", lines=3)
                    with gr.Row():
                        audio_input = gr.Audio(type="filepath", label="Record Audio")
                        image_input = gr.Image(type="filepath", label="Upload Image")
                    submit_button = gr.Button("Submit")
            
            with gr.Row():
                speaker_button = gr.Button("🔊 Play Response")
                audio_output = gr.Audio()
            
            state = gr.State([])
            
            submit_button.click(
                handle_input_modified,
                inputs=[text_input, state, image_input],
                outputs=[chatbot, state, text_input]
            )
            audio_input.change(
                process_audio,
                inputs=[audio_input, state],
                outputs=[text_input, state]
            )
            speaker_button.click(
                play_latest_response,
                inputs=[state],
                outputs=[audio_output]
            )

        with gr.TabItem("Accident Analysis", id="tab-analysis"):
            gr.Markdown("### Accident Analysis")
            
            input_type = gr.Radio(["pdf", "text"], label="Choose input type", value="pdf")
            
            with gr.Row():
                with gr.Column(scale=1, visible=True) as pdf_col:
                    file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                with gr.Column(scale=2, visible=False) as text_col:
                    text_input = gr.Textbox(label="Enter your accident case", lines=5)
            
            analyze_button = gr.Button("Analyze")
            output = gr.Textbox(label="Analysis Result", lines=10)
            
            # Toggle visibility based on input type selection
            def toggle_input_type(choice):
                if choice == "pdf":
                    return gr.update(visible=True), gr.update(visible=False)
                else:
                    return gr.update(visible=False), gr.update(visible=True)
            
            input_type.change(
                toggle_input_type,
                inputs=[input_type],
                outputs=[pdf_col, text_col]
            )
            
            analyze_button.click(
                handle_input,
                inputs=[input_type, file_input, text_input],
                outputs=[output, file_input, text_input]
            )

# Launch the Gradio interface
demo.launch(share=True)
