import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Check if CUDA is available, otherwise use CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Set the appropriate tensor type based on the available device
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Model ID for the OpenAI Whisper model
model_id = "openai/whisper-small"

# Load the pre-trained model with specified configurations
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
)

# Move the model to the appropriate device (CPU or CUDA)
model.to(device)

# Configure model's generation settings
model.generation_config.forced_decoder_ids = None
model.generation_config.language = "en"

# Load the processor which includes both tokenizer and feature extractor
processor = AutoProcessor.from_pretrained(model_id)

# Create a pipeline for automatic speech recognition
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    batch_size=16,
    return_timestamps=True, 
    torch_dtype=torch_dtype,
    device=device
)

# Function to transcribe audio to text using the pipeline
def transcribe_to_text(audio):
    result = pipe(audio)
    return result['text']

# Example usage (commented out)
# audio = "audio.wav"
# result = pipe(audio)
# print(device)
# print(torch_dtype)
# print(pipe)
# print(result['text'])
