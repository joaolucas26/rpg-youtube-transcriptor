import os 
import re

from utils.utils import format_timestamp, timestamp_to_seconds

def merge_logs_from_folder(log_folder, final_transcript_path):
    """
    Finds all '_log.txt' files in the specified folder, reads them,
    and merges them into a single, chronologically sorted file.
    """
    print(f"--- Scanning for log files in '{log_folder}'... ---")
    
    if not os.path.exists(log_folder):
        print(f"❌ Error: Log folder not found at '{log_folder}'")
        return

    # Find all files in the directory that end with '_log.txt'
    log_files = [f for f in os.listdir(log_folder) if f.endswith("_log.txt")]

    if not log_files:
        print(f"No log files found in '{log_folder}'. Nothing to merge.")
        return

    print(f"Found {len(log_files)} log files. Proceeding with merge.")
    all_transcripts = []
    
    for log_filename in log_files:
        log_file_path = os.path.join(log_folder, log_filename)
        print(f"Reading {log_filename}...")
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Use regex to parse the line format "[TIMESTAMP] speaker: text"
                match = re.match(r'\[(.*?)\] (.*?): (.*)', line)
                if match:
                    timestamp_str, speaker, text = match.groups()
                    all_transcripts.append({
                        'speaker': speaker,
                        'start': timestamp_to_seconds(timestamp_str),
                        'text': text.strip()
                    })

    # Sort the combined list chronologically by the start time
    all_transcripts.sort(key=lambda x: x['start'])

    # Write the final, sorted transcript
    with open(final_transcript_path, 'w', encoding='utf-8') as f:
        for entry in all_transcripts:
            start_time_str = format_timestamp(entry['start'])
            f.write(f"[{start_time_str}] {entry['speaker']}: {entry['text']}\n")
            
    print(f"\n✅ Success! Final merged transcript saved to '{final_transcript_path}'")
