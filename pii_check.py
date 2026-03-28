import importlib.util

spec = importlib.util.spec_from_file_location(
    "adapters",
    "/Users/achu9972/Documents/GitHub/ece405-assignment2-data-lmoroz/ece405-assignment2-data/tests/adapters.py"
)
adapters = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapters)

from warcio.archiveiterator import ArchiveIterator

wet = "/Users/achu9972/Documents/GitHub/ece405-assignment2-data-lmoroz/ece405-assignment2-data/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz"

found = 0
with open(wet, 'rb') as f:
    for record in ArchiveIterator(f):
        if record.rec_type == 'conversion':
            text = record.content_stream().read().decode('utf-8', errors='replace')
            if len(text.split()) < 20:
                continue
            url = record.rec_headers.get_header('WARC-Target-URI')
            t1, c1 = adapters.run_mask_emails(text)
            t2, c2 = adapters.run_mask_phone_numbers(t1)
            t3, c3 = adapters.run_mask_ips(t2)
            total = c1 + c2 + c3
            if total > 0:
                print(f'URL: {url}')
                print(f'  emails={c1}, phones={c2}, ips={c3}')
                idx = t3.find('|||')
                if idx >= 0:
                    start = max(0, idx - 60)
                    end = min(len(t3), idx + 40)
                    print(f'  context: ...{t3[start:end]}...')
                print()
                found += 1
        if found >= 20:
            break
