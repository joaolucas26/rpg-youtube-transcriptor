import subprocess
import torch
from pydub import AudioSegment
import numpy as np


def load_silero_vad_model():
    print("Loading Silero VAD model...")
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
    )
    get_speech_timestamps, _, _, _, _ = utils
    print("Silero VAD model loaded.")
    return vad_model, get_speech_timestamps


def load_audio_16k_mono(audio_path: str) -> AudioSegment:
    """
    Decodifica o áudio diretamente para PCM 16kHz Mono 16-bit via ffmpeg,
    evitando o limite de 4GB do cabeçalho WAV no pydub em sessões longas.
    """
    ffmpeg_cmd = [
        AudioSegment.converter,
        "-nostdin",
        "-y",
        "-i", audio_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-",
    ]
    process = subprocess.Popen(
        ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    raw_pcm, stderr = process.communicate()
    if process.returncode != 0 or not raw_pcm:
        raise RuntimeError(
            f"Falha ao decodificar áudio no ffmpeg: {stderr.decode(errors='ignore')}"
        )

    return AudioSegment(
        data=raw_pcm,
        sample_width=2,
        frame_rate=16000,
        channels=1,
    )


def segment_audio_with_vad(audio_path, vad_model, get_speech_timestamps):
    try:
        print(f"Carregando áudio em 16kHz Mono: '{audio_path}'...")
        audio = load_audio_16k_mono(audio_path)
        audio_tensor = (
            torch.from_numpy(np.array(audio.get_array_of_samples())).float()
            / 32768.0
        )

        print("Applying VAD to session audio...")
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            vad_model,
            sampling_rate=16000,
            threshold=0.5,                           # Sensibilidade de voz (0.4 a 0.6). Aumente se pegar música de fundo
            min_speech_duration_ms=500,              # Ignora barulhos menores que 0.5s logo na origem (padrão: 250)
            max_speech_duration_s=8,                 # Limita blocos a no máx 8s para não misturar 2 pessoas na mesma fala
            min_silence_duration_ms=300,             # Corta quando houver 0.3s de pausa (separa perguntas e respostas rápidas)
            speech_pad_ms=150,                       # Margem extra nas pontas para não engolir a 1ª/última sílaba (padrão: 30)
        )
        del audio_tensor
        print(f"Found {len(speech_timestamps)} speech segments.")

        return audio, speech_timestamps
    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        return None, []
