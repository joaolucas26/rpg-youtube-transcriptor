import os
import yt_dlp


def download_youtube_audio(youtube_url: str, output_folder: str = "dnd_audio") -> str:
    """
    Recebe um link do YouTube, baixa a faixa de áudio na pasta especificada
    e retorna o caminho completo do arquivo salvo.
    """
    os.makedirs(output_folder, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_folder, "%(title).80s [%(id)s].%(ext)s"),
        "noplaylist": True,
        "quiet": False,
    }

    print(f"--- Baixando áudio do YouTube: {youtube_url} ---")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        audio_path = ydl.prepare_filename(info)

    print(f"✅ Áudio salvo em: '{audio_path}'")
    return audio_path

