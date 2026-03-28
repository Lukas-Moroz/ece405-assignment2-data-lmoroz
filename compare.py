import sys

import importlib.util
spec = importlib.util.spec_from_file_location(
    "adapters",
    "/Users/achu9972/Documents/GitHub/ece405-assignment2-data-lmoroz/ece405-assignment2-data/tests/adapters.py"
)
adapters = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapters)

from warcio.archiveiterator import ArchiveIterator

warc = "CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
wet = "CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz"

with open(warc, 'rb') as f:
    for record in ArchiveIterator(f):
        if record.rec_type == 'response':
            html_bytes = record.content_stream().read()
            your_extraction = adapters.run_extract_text_from_html_bytes(html_bytes)
            break

with open(wet, 'rb') as f:
    for record in ArchiveIterator(f):
        if record.rec_type == 'conversion':
            wet_extraction = record.content_stream().read().decode('utf-8', errors='replace')
            break

print('=== YOUR EXTRACTION (Resiliparse) ===')
print(your_extraction[:1500])
print()
print('=== COMMON CRAWL WET EXTRACTION ===')
print(wet_extraction[:1500])