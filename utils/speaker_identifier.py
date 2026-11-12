import os
import torch
import numpy as np
import torchaudio
from pydub import AudioSegment


if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

def load_speaker_model():
    print("Loading Speaker Recognition model (SpeechBrain)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: GPU not found. Process will be very slow.")

    speaker_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY  # Evita erro de permissão de atalho (symlink) no Windows
    )
    return speaker_model

def load_character_profiles(speaker_model, samples_folder="sample_audios"):
    print(f"Loading character voice profiles from '{samples_folder}'...")
    character_profiles = {}

    with torch.no_grad():
        for filename in os.listdir(samples_folder):
            if not filename.endswith((".ogg", ".wav", ".flac", ".mp3")):
                continue

            character_name = os.path.splitext(filename)[0]
            filepath = os.path.join(samples_folder, filename)

            audio = AudioSegment.from_file(filepath)
            audio_16k = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            audio_tensor = torch.from_numpy(np.array(audio_16k.get_array_of_samples())).float() / 32768.0

            embedding = speaker_model.encode_batch(audio_tensor.unsqueeze(0)).squeeze()
            embedding = torch.nn.functional.normalize(embedding, dim=0)

            character_profiles[character_name] = embedding
            print(f"Perfil carregado: {character_name}")

    return character_profiles


def identify_speaker(speech_chunk, speaker_model, character_profiles):
    with torch.no_grad():
        audio_16k = speech_chunk.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio_tensor = torch.from_numpy(np.array(audio_16k.get_array_of_samples())).float() / 32768.0
        if len(audio_tensor) < 3200:
            repeats = (3200 // max(len(audio_tensor), 1)) + 1
            audio_tensor = audio_tensor.repeat(repeats)

        chunk_embedding = speaker_model.encode_batch(audio_tensor.unsqueeze(0)).squeeze()
        chunk_embedding = torch.nn.functional.normalize(chunk_embedding, dim=0)

        best_speaker = "desconhecido"
        best_score = -1.0
        for character_name, profile_embedding in character_profiles.items():
            score = torch.dot(chunk_embedding, profile_embedding).item()
            if score > best_score:
                best_score = score
                best_speaker = character_name
        return best_speaker