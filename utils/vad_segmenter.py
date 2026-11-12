
import torch
from pydub import AudioSegment
import numpy as np



def load_silero_vad_model():
    print("Loading Silero VAD model...")
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    (get_speech_timestamps, _, _, _, _) = utils
    print("Silero VAD model loaded.")
    return vad_model, get_speech_timestamps


def segment_audio_with_vad(audio_path, vad_model, get_speech_timestamps):
    try:
        audio = AudioSegment.from_file(audio_path)
        audio_for_vad = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio_tensor = torch.from_numpy(np.array(audio_for_vad.get_array_of_samples())).float() / 32768.0

        print("Applying VAD to session audio...")
        speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=16000)
        del audio_tensor
        print(f"Found {len(speech_timestamps)} speech segments.")

        return audio, speech_timestamps
    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        return None, []