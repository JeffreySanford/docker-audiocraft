from importlib.machinery import SourceFileLoader
mod=SourceFileLoader('u','/workspace/utils.py').load_module()
title, lyrics, styles = mod.parse_song_file('/workspace/lyrics_terraform_my_heart.txt')
meta = styles.get('_meta', {})
dur = meta.get('duration','')
tempo = meta.get('tempo','')
if dur:
    try:
        dur_i = int(''.join(ch for ch in dur if ch.isdigit()))
    except Exception:
        dur_i = 60
else:
    dur_i = 60
# Compose explicit prompt
lines = []
lines.append(f"Sing the following lyrics clearly as a rock song (tempo: {tempo or '120bpm'}, duration target: {dur_i}s). Use lead vocal, electric guitar, bass and drums. Emphasize melody and chorus.")
lines.append('\nLYRICS:\n' + (lyrics.strip() if lyrics else ''))
if styles.get('medium'):
    lines.append('\nStyle: ' + styles.get('medium'))
lines.append('\nVocal: clear, lead vocal; sing the lyrics verbatim with emotional phrasing.')
print('\n\n'.join(lines))
