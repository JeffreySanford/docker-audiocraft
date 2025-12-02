#!/usr/bin/env python3
import os
from utils import parse_song_file

LYRICS_FILE = os.path.join(os.path.dirname(__file__), 'lyrics_terraform_my_heart.txt')

if __name__ == '__main__':
    title, lyrics, styles = parse_song_file(LYRICS_FILE)
    base = lyrics.strip() if lyrics else ''
    style = ''
    if isinstance(styles, dict):
        style = styles.get('small', '') or styles.get('default', '')
    extra = 'female jazz singer, clean English pronunciation, relaxed late-night club ambience'
    parts = [p for p in [base, style, extra] if p]
    prompt = '\n\n'.join(parts)
    print(prompt)
