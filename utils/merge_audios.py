import os
import re

from utils.utils import format_timestamp, timestamp_to_seconds


def _append_text_segment(current_text: str, next_text: str) -> str:
    """
    Concatena falas consecutivas do mesmo speaker de forma fluida:
    - Remove '...' no final do trecho anterior se a frase continua;
    - Remove ponto final '.' se o próximo trecho começar com letra minúscula.
    """
    prev = current_text.strip()
    nxt = next_text.strip()
    if not prev:
        return nxt
    if not nxt:
        return prev

    if prev.endswith("..."):
        prev = prev[:-3].rstrip()
    elif prev.endswith(".") and nxt[0].islower():
        prev = prev[:-1].rstrip()

    return f"{prev} {nxt}"


def merge_logs_from_folder(log_folder, final_transcript_path):
    """
    Finds all '_log.txt' files in the specified folder, reads them,
    sorts them chronologically, and merges consecutive lines spoken by
    the same speaker into a single entry.
    """
    print(f"--- Scanning for log files in '{log_folder}'... ---")

    if not os.path.exists(log_folder):
        print(f"❌ Error: Log folder not found at '{log_folder}'")
        return

    log_files = [f for f in os.listdir(log_folder) if f.endswith("_log.txt")]

    if not log_files:
        print(f"No log files found in '{log_folder}'. Nothing to merge.")
        return

    print(f"Found {len(log_files)} log files. Proceeding with merge.")
    all_transcripts = []

    for log_filename in log_files:
        log_file_path = os.path.join(log_folder, log_filename)
        print(f"Reading {log_filename}...")
        with open(log_file_path, "r", encoding="utf-8") as f:
            for line in f:
                match = re.match(r"\[(.*?)\] (.*?): (.*)", line)
                if match:
                    timestamp_str, speaker, text = match.groups()
                    all_transcripts.append(
                        {
                            "speaker": speaker,
                            "start": timestamp_to_seconds(timestamp_str),
                            "text": text.strip(),
                        }
                    )

    # 1. Ordenar cronologicamente pelo timestamp de início
    all_transcripts.sort(key=lambda x: x["start"])

    # 2. Agrupar falas consecutivas do mesmo speaker
    merged_transcripts = []
    for entry in all_transcripts:
        if not entry["text"]:
            continue
        if merged_transcripts and merged_transcripts[-1]["speaker"] == entry["speaker"]:
            merged_transcripts[-1]["text"] = _append_text_segment(
                merged_transcripts[-1]["text"], entry["text"]
            )
        else:
            merged_transcripts.append(
                {
                    "speaker": entry["speaker"],
                    "start": entry["start"],
                    "text": entry["text"],
                }
            )

    # 3. Salvar a transcrição final consolidada
    with open(final_transcript_path, "w", encoding="utf-8") as f:
        for entry in merged_transcripts:
            start_time_str = format_timestamp(entry["start"])
            f.write(f"[{start_time_str}] {entry['speaker']}: {entry['text']}\n")

    print(
        f"\n✅ Success! Final merged transcript saved to '{final_transcript_path}' "
        f"({len(all_transcripts)} trechos agrupados em {len(merged_transcripts)} falas)."
    )


if __name__ == "__main__":
    OUTPUTS_DIR = "outputs"
    if os.path.exists(OUTPUTS_DIR):
        for folder in os.listdir(OUTPUTS_DIR):
            video_dir = os.path.join(OUTPUTS_DIR, folder)
            transcriptions_dir = os.path.join(video_dir, "transcriptions")
            target_dir = transcriptions_dir if os.path.isdir(transcriptions_dir) else video_dir
            if os.path.isdir(target_dir):
                merge_logs_from_folder(
                    target_dir,
                    os.path.join(target_dir, "final_session_transcript.txt"),
                )


if __name__ == "__main__":
    OUTPUTS_DIR = "outputs"
    if os.path.exists(OUTPUTS_DIR):
        for folder in os.listdir(OUTPUTS_DIR):
            video_dir = os.path.join(OUTPUTS_DIR, folder)
            transcriptions_dir = os.path.join(video_dir, "transcriptions")
            target_dir = transcriptions_dir if os.path.isdir(transcriptions_dir) else video_dir
            if os.path.isdir(target_dir):
                merge_logs_from_folder(
                    target_dir,
                    os.path.join(target_dir, "final_session_transcript.txt"),
                )
