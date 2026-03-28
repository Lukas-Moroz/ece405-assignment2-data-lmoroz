import gzip, random, requests, importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed

spec = importlib.util.spec_from_file_location('adapters', './tests/adapters.py')
adapters = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapters)

print('Loading URLs...')
with gzip.open('enwiki-20240420-extracted_urls.txt.gz', 'rt', errors='replace') as f:
    urls = [line.strip() for line in f if line.strip().startswith('http')]

# Upgraded: Sample 20,000 URLs to ensure we have a big enough pool
sampled = random.sample(urls, min(20000, len(urls)))
print(f'Sampled {len(sampled)} URLs')

results = []
TARGET_POSITIVES = 1000 # The goal we want to hit

def fetch(url):
    try:
        r = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
        if r.status_code == 200:
            text = adapters.run_extract_text_from_html_bytes(r.content)
            if text and len(text.split()) >= 50:
                lang, score = adapters.run_identify_language(text[:500])
                if lang == 'en' and score > 0.8:
                    return text[:500].replace('\n', ' ')
    except:
        pass
    return None

print(f'Fetching pages until we hit {TARGET_POSITIVES} positives...')
# Upgraded: 50 workers for much faster network requests
with ThreadPoolExecutor(max_workers=50) as ex:
    futures = {ex.submit(fetch, url): url for url in sampled}
    for i, f in enumerate(as_completed(futures)):
        result = f.result()
        if result:
            results.append(result)
            # Print a status update every 50 successes
            if len(results) % 50 == 0:
                print(f'  Collected {len(results)}/{TARGET_POSITIVES} positives...')
                
        # Upgraded: Stop early once we hit our target
        if len(results) >= TARGET_POSITIVES:
            print("Target reached! Wrapping up...")
            # Cancel all pending futures that haven't started yet!
            ex.shutdown(wait=False, cancel_futures=True)
            break

with open('quality_positives.txt', 'w') as f:
    for doc in results:
        f.write(doc + '\n')
print(f'Wrote {len(results)} positives to quality_positives.txt')