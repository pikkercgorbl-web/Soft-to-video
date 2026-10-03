"""Read-only client diagnostics; never synthesizes speech or calls Gemini."""
import argparse
import re
import socket
from pathlib import Path
from urllib.parse import urlsplit
import os

import download_pexels as pipeline


def inspect_saved_config(path: Path, language: str) -> list[str]:
    if not path.is_file():
        return [f'Saved config absent: {path}']
    text = path.read_text(encoding='utf-8')
    match = re.search(r'^\s*repo_id:\s*([^\n#]+)', text, re.MULTILINE)
    model = match.group(1).strip().strip("\"'") if match else 'unknown'
    messages = [f'Saved config model: {model} (NOT confirmation of the running model)']
    if 'turbo' in model.lower() and language.lower() != 'en':
        messages.append('WARNING: saved Turbo configuration and non-English client language. '
                        'Check the actual model in the server startup log.')
    return messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--offline', action='store_true', help='Skip TCP connection check')
    args = parser.parse_args()
    pipeline.load_settings()
    address = urlsplit(pipeline.CHATTERBOX_TTS_URL)
    print(f'Client language: {pipeline.TTS_LANGUAGE}')
    print(f'Voice: {pipeline.TTS_VOICE_FILE}; mode: {pipeline.TTS_VOICE_MODE}')
    print(f'Declared model: {os.getenv("TTS_MODEL_ID", "unknown")} (user-supplied)')
    print(f'Cache revision: {os.getenv("TTS_CACHE_REVISION", "1")}')
    print(f'Temperature: {pipeline.TTS_TEMPERATURE}; exaggeration: {pipeline.TTS_EXAGGERATION}; '
          f'CFG: {pipeline.TTS_CFG_WEIGHT}; chunk size: {pipeline.TTS_CHUNK_SIZE}')
    for message in inspect_saved_config(Path(args.config), pipeline.TTS_LANGUAGE):
        print(message)
    print('Keep server startup model/version and a RAW WAV for the listening comparison.')
    if args.offline:
        return 0
    if address.scheme not in {'http', 'https'} or not address.hostname:
        print('Invalid TTS endpoint: expected an HTTP(S) URL.')
        return 1
    try:
        port = address.port or (443 if address.scheme == 'https' else 80)
        with socket.create_connection((address.hostname, port), timeout=5):
            print(f'TCP connection OK: {address.hostname}:{port}. API/model not verified.')
        return 0
    except (OSError, ValueError) as error:
        print(f'TTS connection unavailable: {type(error).__name__}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
