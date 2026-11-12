import os
import re
import yt_dlp


def _extract_video_id(youtube_url: str) -> str | None:
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", youtube_url)
    return match.group(1) if match else None


def download_youtube_audio(youtube_url: str, output_folder: str = "inputs/youtube_original_audio") -> str:
    """
    Recebe um link do YouTube, verifica se o áudio já foi baixado na pasta especificada;
    caso contrário, faz o download e retorna o caminho completo do arquivo salvo.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Verifica se já existe um arquivo com o mesmo ID do vídeo na pasta para evitar novo download
    video_id = _extract_video_id(youtube_url)
    if video_id:
        for filename in os.listdir(output_folder):
            if f"[{video_id}]" in filename and not filename.endswith(".part"):
                existing_path = os.path.join(output_folder, filename)
                print(f"✅ Áudio já existe localmente, pulando download: '{existing_path}'")
                return existing_path

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
