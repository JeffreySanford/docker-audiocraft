from importlib.machinery import SourceFileLoader
mod = SourceFileLoader('u', '/workspace/utils.py').load_module()
title, lyrics, styles = mod.parse_song_file('/workspace/lyrics_terraform_my_heart.txt')
meta = styles.get('_meta', {})
# get a short chorus snippet (try to find [Chorus] markers)
chorus = ''
if 'chorus' in lyrics.lower():
    import re
    m = re.search(r'(?i)\[chorus[^\]]*\]([\s\S]{1,800})', lyrics)
    if m:
        chorus = m.group(1).strip()
if not chorus:
    # fallback to first 40-60 words
    words = (lyrics or '').split()
    chorus = ' '.join(words[:60])

variants = {}
# Variant A: explicit sing verbatim, chorus-only
variants['A'] = 'SING THE FOLLOWING VERBATIM:\nLYRICS:\n' + chorus + '\n\nVocal: lead vocal, rock; sing exactly the words with a clear melody.'
# Variant B: instruction-first then full lyrics
variants['B'] = 'Sing the following lyrics clearly as a rock song (tempo: {tempo}). Lead vocal, electric guitar, bass, drums.\n\nLYRICS:\n{lyrics}\n\nVocal: clear, sung, strong presence; deliver melody with rhythm and emotion.'.format(tempo=meta.get('tempo','120bpm'), lyrics=lyrics)
# Variant C: lyric-labeled with imperative
variants['C'] = 'PERFORM THE FOLLOWING LYRIC BLOCK AS A SUNG LEAD VOCAL. DO NOT SPEAK OR HUM — SING THE WORDS.\n\nLYRICS:\n' + chorus + '\n\nVocal: strong rock lead, sing the chorus verbatim.'

# Write out files
for k, v in variants.items():
    path = f'/tmp/prompt_variant_{k}.txt'
    with open(path, 'w', encoding='utf8') as fh:
        fh.write(v)
    print('WROTE', path)
