import os
import gc
import torch
import static_ffmpeg
static_ffmpeg.add_paths()



from utils.utils import format_timestamp
from utils.youtube_downloader import download_youtube_audio
from utils.vad_segmenter import load_silero_vad_model, segment_audio_with_vad
from utils.speaker_identifier import load_speaker_model, load_character_profiles, identify_speaker
from utils.transcriber import load_whisper_model, transcribe_chunk
from utils.merge_audios import merge_logs_from_folder


YOUTUBE_URL = ""  # Coloque o link do YouTube aqui
AUDIO_FOLDER = "dnd_audio"
YOUTUBE_AUDIO_FOLDER = "youtube_original_audio"
SAMPLES_FOLDER = "sample_audios"
OUTPUT_LOG_FOLDER = "transcription_player_logs"
FINAL_TRANSCRIPT_PATH = "final_session_transcript.txt"
WHISPER_MODEL = "medium"



def run_transcription():
    print("--- Starting Full Transcription Process ---")
    os.makedirs(OUTPUT_LOG_FOLDER, exist_ok=True)
    os.makedirs(YOUTUBE_AUDIO_FOLDER, exist_ok=True)

    url = YOUTUBE_URL or input("Cole a URL do vídeo do YouTube: ").strip()
    session_audio_path = download_youtube_audio(url, AUDIO_FOLDER)

    vad_model, get_speech_timestamps = load_silero_vad_model()
    speaker_model = load_speaker_model()
    character_profiles = load_character_profiles(speaker_model, SAMPLES_FOLDER)
    whisper_model = load_whisper_model(WHISPER_MODEL)


    audio, speech_timestamps = segment_audio_with_vad(
        session_audio_path, vad_model, get_speech_timestamps
    )

    if not speech_timestamps:
        print("⚠️ Nenhuma fala detectada no áudio.")
        return

    total_chunks = len(speech_timestamps)
    for i, ts in enumerate(speech_timestamps):
        start_ms = ts["start"] / 16
        end_ms = ts["end"] / 16
        speech_chunk = audio[start_ms:end_ms]

        # Quem está falando?
        speaker_name = identify_speaker(speech_chunk, speaker_model, character_profiles)
        print(f"[{i+1}/{total_chunks}] Speaker detectado: {speaker_name} — Transcrevendo...")

        # O que foi dito?
        chunk_entries = transcribe_chunk(speech_chunk, start_ms, speaker_name, whisper_model)

        # Salvar imediatamente no log individual do personagem
        if chunk_entries:
            log_filename = os.path.join(OUTPUT_LOG_FOLDER, f"{speaker_name}_log.txt")
            with open(log_filename, "a", encoding="utf-8") as log_file:
                for entry in chunk_entries:
                    start_time = format_timestamp(entry["start"])
                    log_file.write(f"[{start_time}] {entry['speaker']}: {entry['text']}\n")

    # Limpeza de memória
    del audio
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Unificar todos os logs dos jogadores em ordem cronológica
    merge_logs_from_folder(OUTPUT_LOG_FOLDER, FINAL_TRANSCRIPT_PATH)


if __name__ == "__main__":
    run_transcription()