import os
import torch
import whisper

def load_whisper_model(model_name="medium"):
    print(f"Loading Whisper model: '{model_name}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: GPU not found. Process will be very slow.")
    whisper_model = whisper.load_model(model_name, device=device)
    print(f"Whisper model loaded on '{device}'.")
    return whisper_model


def transcribe_chunk(speech_chunk, start_ms, speaker_name, whisper_model):
    temp_chunk_path = "temp_chunk.wav"
    speech_chunk.export(temp_chunk_path, format="wav")
    result = whisper_model.transcribe(
        temp_chunk_path,
        language="pt",
        fp16=torch.cuda.is_available()
    )

    chunk_transcripts = []
    for segment in result["segments"]:
        text = segment["text"].strip()
        if text:
            corrected_start = (start_ms / 1000.0) + segment["start"]
            chunk_transcripts.append({
                "speaker": speaker_name,
                "start": corrected_start,
                "text": text
            })
            
    if os.path.exists(temp_chunk_path):
        os.remove(temp_chunk_path)
    return chunk_transcripts