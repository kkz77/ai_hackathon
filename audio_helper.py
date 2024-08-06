import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-small"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id,torch_dtype = torch_dtype, 
                                                  low_cpu_mem_usage = True, use_safetensors = True)                   

model.to(device)
model.generation_config.forced_decoder_ids = None
model.generation_config.language = "en"
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens = 128,
    chunk_length_s = 15,
    batch_size = 16,
    return_timestamps = True, 
    torch_dtype= torch_dtype,
    device= device
)

def transcribe_to_text(audio):
    result = pipe(audio)
    return result['text']

# audio = "audio.wav"
# result = pipe(audio)
# print(device)
# print(torch_dtype)
# print(pipe)
# print(result['text'])