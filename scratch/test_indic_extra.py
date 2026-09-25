import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from indic_transliteration import sanscript
import re

text_gur = 'ਲੋਟਸ ਕੰਸਲਟਿੰਗ ਲਿਮਟਿਡ'
res_gur = sanscript.transliterate(text_gur, sanscript.GURMUKHI, sanscript.ITRANS)
print("Gurmukhi:", res_gur, "Prefix4:", re.sub(r'[^a-zA-Z0-9]', '', res_gur).lower()[:4])

text_ben = 'স্টার অ্যাগ্রો প্রাইভেট লিমিটেড'
res_ben = sanscript.transliterate('স্টার অ্যাগ্রো প্রাইভেট লিমিটেড', sanscript.BENGALI, sanscript.ITRANS)
print("Bengali:", res_ben, "Prefix4:", re.sub(r'[^a-zA-Z0-9]', '', res_ben).lower()[:4])
