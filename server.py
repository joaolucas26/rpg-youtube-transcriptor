import os
import gc
import torch
import static_ffmpeg
static_ffmpeg.add_paths()

from utils.utils import format_timestamp
from utils.youtube_downloader import download_youtube_audio
from utils.vad_segmenter import load_silero_vad_model, segment_audio_with_vad
from utils.speaker_identifier import (
    load_speaker_model,
    load_character_profiles,
    warmup_session_profiles,
    identify_speaker,
)
from utils.transcriber import load_whisper_model, transcribe_chunk
from utils.merge_audios import merge_logs_from_folder
from utils.analytics import SessionAnalytics


YOUTUBE_URL = ""  # Coloque o link do YouTube aqui
INPUTS_FOLDER = "inputs"
OUTPUTS_FOLDER = "outputs"

YOUTUBE_AUDIO_FOLDER = os.path.join(INPUTS_FOLDER, "youtube_original_audio")
SAMPLES_FOLDER = os.path.join(INPUTS_FOLDER, "sample_audios")
INITIAL_PROMPT_PATH = os.path.join(INPUTS_FOLDER, "initial_prompt.txt")
WHISPER_MODEL = "turbo"
MAX_SPEAKERS = 6
USE_VOICE_SAMPLES = True  # Defina False para testar o modo sem upload de samples (100% automático)


def load_initial_prompt(prompt_path: str) -> str:
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
            print(f"Initial prompt carregado de '{prompt_path}'.")
            return prompt
    print(f"⚠️ Arquivo '{prompt_path}' não encontrado. Seguindo sem initial_prompt.")
    return ""


def run_transcription():
    print("--- Starting Full Transcription Process ---")
    os.makedirs(YOUTUBE_AUDIO_FOLDER, exist_ok=True)
    os.makedirs(SAMPLES_FOLDER, exist_ok=True)
    os.makedirs(OUTPUTS_FOLDER, exist_ok=True)

    initial_prompt = load_initial_prompt(INITIAL_PROMPT_PATH)

    url = YOUTUBE_URL or input("Cole a URL do vídeo do YouTube: ").strip()
    session_audio_path = download_youtube_audio(url, YOUTUBE_AUDIO_FOLDER)

    video_folder_name = os.path.splitext(os.path.basename(session_audio_path))[0]
    video_output_folder = os.path.join(OUTPUTS_FOLDER, video_folder_name)
    os.makedirs(video_output_folder, exist_ok=True)
    for old_file in os.listdir(video_output_folder):
        if old_file.endswith((".txt", ".csv", ".json")):
            os.remove(os.path.join(video_output_folder, old_file))
    final_transcript_path = os.path.join(video_output_folder, "final_session_transcript.txt")

    vad_model, get_speech_timestamps = load_silero_vad_model()
    speaker_model = load_speaker_model()
    character_profiles = load_character_profiles(
        speaker_model, SAMPLES_FOLDER, use_samples=USE_VOICE_SAMPLES
    )
    whisper_model = load_whisper_model(WHISPER_MODEL)

    audio, speech_timestamps = segment_audio_with_vad(
        session_audio_path, vad_model, get_speech_timestamps
    )

    if not speech_timestamps:
        print("⚠️ Nenhuma fala detectada no áudio.")
        return

    # Warm-up: alinha samples externos (ex: WhatsApp) ao microfone da sessão e/ou
    # descobre automaticamente os clusters de voz caso o cliente não tenha enviado samples
    character_profiles = warmup_session_profiles(
        audio=audio,
        speech_timestamps=speech_timestamps,
        speaker_model=speaker_model,
        character_profiles=character_profiles,
        max_speakers=MAX_SPEAKERS,
    )

    analytics = SessionAnalytics(samples_folder=SAMPLES_FOLDER)
    analytics.inspect_samples(character_profiles)

    total_chunks = len(speech_timestamps)
    try:
        for i, ts in enumerate(speech_timestamps):
            start_ms = ts["start"] / 16
            end_ms = ts["end"] / 16
            speech_chunk = audio[start_ms:end_ms]

            # Quem está falando?
            speaker_name = identify_speaker(
                speech_chunk,
                speaker_model,
                character_profiles,
                max_speakers=MAX_SPEAKERS,
                analytics=analytics,
                chunk_index=i + 1,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            if speaker_name == "ignored":
                print(f"[{i+1}/{total_chunks}] Trecho curto/ruído ignorado.")
                continue

            last_rec = analytics.records[-1]
            print(
                f"[{i+1}/{total_chunks}] Speaker: {speaker_name} "
                f"(score: {last_rec['best_score']:.2f} | 2º: {last_rec['second_best_speaker']} {last_rec['second_best_score']:.2f}"
                f"{' | 🎯 calibrado' if last_rec['was_adapted'] else ''}) — Transcrevendo..."
            )

            # O que foi dito?
            chunk_entries = transcribe_chunk(
                speech_chunk, start_ms, speaker_name, whisper_model, initial_prompt
            )

            # Salvar imediatamente no log individual do personagem dentro da pasta do vídeo
            if chunk_entries:
                log_filename = os.path.join(video_output_folder, f"{speaker_name}_log.txt")
                with open(log_filename, "a", encoding="utf-8") as log_file:
                    for entry in chunk_entries:
                        start_time = format_timestamp(entry["start"])
                        log_file.write(f"[{start_time}] {entry['speaker']}: {entry['text']}\n")
    except KeyboardInterrupt:
        print("\n⏹️ Execução interrompida pelo usuário! Salvando transcrição parcial e relatório de analytics...")
    finally:
        # Limpeza de memória
        del audio
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Gerar relatórios de Analytics e unificar logs
        analytics.save_reports(video_output_folder, character_profiles)
        merge_logs_from_folder(video_output_folder, final_transcript_path)


if __name__ == "__main__":
    run_transcription()
