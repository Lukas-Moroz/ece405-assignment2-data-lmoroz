import importlib.util

spec = importlib.util.spec_from_file_location(
    "adapters",
    "/Users/achu9972/Documents/GitHub/ece405-assignment2-data-lmoroz/ece405-assignment2-data/tests/adapters.py"
)
adapters = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapters)

from warcio.archiveiterator import ArchiveIterator

wet = "CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz"

count = 0
flagged = 0
with open(wet, 'rb') as f:
    for record in ArchiveIterator(f):
        if record.rec_type == 'conversion':
            text = record.content_stream().read().decode('utf-8', errors='replace')
            if len(text.split()) < 20:
                continue
            url = record.rec_headers.get_header('WARC-Target-URI')
            nsfw_label, nsfw_score = adapters.run_classify_nsfw(text[:1000])
            tox_label, tox_score = adapters.run_classify_toxic_speech(text[:1000])
            is_flagged = nsfw_label != 'non-nsfw' or tox_label != 'non-toxic'
            if is_flagged:
                flagged += 1
            print(f"{count+1:2}. nsfw={nsfw_label}({nsfw_score:.2f}) toxic={tox_label}({tox_score:.2f}) {url}")
            count += 1
        if count >= 20:
            break

print(f"\nFlagged as harmful: {flagged}/20")
