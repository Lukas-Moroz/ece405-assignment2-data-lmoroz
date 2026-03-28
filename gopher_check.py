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
passed = 0
with open(wet, 'rb') as f:
    for record in ArchiveIterator(f):
        if record.rec_type == 'conversion':
            text = record.content_stream().read().decode('utf-8', errors='replace')
            if len(text.split()) < 5:
                continue
            url = record.rec_headers.get_header('WARC-Target-URI')
            result = adapters.run_gopher_quality_filter(text)
            words = text.split()
            n = len(words)
            avg = sum(len(w) for w in words) / n if n else 0
            alpha = sum(1 for w in words if any(c.isalpha() for c in w))
            alpha_ratio = alpha / n if n else 0
            passed += result
            print(f"{count+1:2}. [{'PASS' if result else 'FAIL'}] words={n} avg_len={avg:.1f} alpha={alpha_ratio:.2f} {url}")
            count += 1
        if count >= 20:
            break

print(f"\nPassed: {passed}/20")
