

from datetime import timedelta


def format_timestamp(seconds):
    """Formats seconds into HH:MM:SS,ms format."""
    delta = timedelta(seconds=seconds)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = delta.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def timestamp_to_seconds(ts_str):
    """Converts HH:MM:SS,ms string back to total seconds."""
    try:
        h, m, s_ms = ts_str.split(':')
        s, ms = s_ms.split(',')
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
    except (ValueError, AttributeError):
        return 0.0