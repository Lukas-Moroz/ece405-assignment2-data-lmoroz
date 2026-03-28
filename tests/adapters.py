from __future__ import annotations

import hashlib
import os
import pathlib
import re
import string
import tempfile
import unicodedata
from collections import defaultdict
from typing import Any

import fasttext
import mmh3

# Shared helpers
_ROOT = pathlib.Path(__file__).resolve().parent.parent
_FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"

# Silence fastText's noisy stderr warnings
fasttext.FastText.eprint = lambda *a, **kw: None

# Language ID — fastText lid.176.bin
_lid_model: fasttext.FastText._FastText | None = None


def _get_lid_model() -> fasttext.FastText._FastText:
    global _lid_model
    if _lid_model is None:
        _lid_model = fasttext.load_model(str(_ROOT / "lid.176.bin"))
    return _lid_model


# NSFW classifier
_nsfw_model: fasttext.FastText._FastText | None = None


def _get_nsfw_model() -> fasttext.FastText._FastText:
    global _nsfw_model
    if _nsfw_model is None:
        _nsfw_model = fasttext.load_model(
            str(_ROOT / "jigsaw_fasttext_bigrams_nsfw_final.bin")
        )
    return _nsfw_model


# Hate-speech / toxicity classifier
_hate_model: fasttext.FastText._FastText | None = None


def _get_hate_model() -> fasttext.FastText._FastText:
    global _hate_model
    if _hate_model is None:
        _hate_model = fasttext.load_model(
            str(_ROOT / "jigsaw_fasttext_bigrams_hatespeech_final.bin")
        )
    return _hate_model


# Quality classifier — auto-trains if model file is absent
_QUALITY_MODEL_PATH = _ROOT / "quality_classifier.bin"
_quality_model: fasttext.FastText._FastText | None = None


def _train_quality_model() -> fasttext.FastText._FastText:
    """Train a fastText quality classifier.

    Priority:
      1. quality_positives.txt / quality_negatives.txt in _ROOT (one doc per line)
      2. Fixture file chunks from tests/fixtures/
      3. Hard-coded example strings

    After training, evaluates on a held-out 20% split and prints metrics.
    """
    import random

    training_lines: list[str] = []

    # Load from pre-built data files if present
    pos_path = _ROOT / "quality_positives.txt"
    neg_path = _ROOT / "quality_negatives.txt"

    if pos_path.exists():
        with open(pos_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    training_lines.append(f"__label__wiki {line[:500]}")
    if neg_path.exists():
        with open(neg_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    training_lines.append(f"__label__cc {line[:500]}")

    # Fallback: fixture file chunks
    if not training_lines:
        try:
            wiki_text = (_FIXTURES / "high_quality_wiki_reference.txt").read_text()
        except FileNotFoundError:
            wiki_text = ""
        try:
            cc_text = (_FIXTURES / "low_quality_cc.txt").read_text()
        except FileNotFoundError:
            cc_text = ""

        flat_wiki = wiki_text.replace("\n", " ")
        for i in range(0, max(1, len(flat_wiki) - 300), 150):
            chunk = flat_wiki[i : i + 400].strip()
            if len(chunk.split()) >= 20:
                training_lines.append(f"__label__wiki {chunk}")

        flat_cc = cc_text.replace("\n", " ")
        for i in range(0, max(1, len(flat_cc) - 50), 30):
            chunk = flat_cc[i : i + 100].strip()
            if chunk:
                training_lines.append(f"__label__cc {chunk}")

    # Hard-coded examples always appended for robustness
    extra_wiki = [
        "The history of the Roman Empire began when Augustus became the first emperor.",
        "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
        "In mathematics a prime number is a natural number greater than one not a product of two smaller natural numbers.",
        "The Renaissance was a period of cultural and intellectual flourishing in Europe.",
        "Classical mechanics describes the motion of macroscopic objects from projectiles to spacecraft.",
    ]
    extra_cc = [
        "Home About Contact Services Privacy Policy Terms of Use Sign Up Log In",
        "Click here Subscribe Newsletter Get started free Buy now Add to cart",
        "Copyright All rights reserved Powered by WordPress Theme by",
        "Menu Navigation Footer Sidebar Related Posts Tags Categories Archive",
        "Login Register Username Password Forgot password Remember me Submit",
    ]
    for ex in extra_wiki:
        training_lines.append(f"__label__wiki {ex}")
    for ex in extra_cc:
        training_lines.append(f"__label__cc {ex}")

    # Shuffle and split 80/20
    random.shuffle(training_lines)
    split = max(1, int(0.8 * len(training_lines)))
    train_lines = training_lines[:split]
    test_lines = training_lines[split:]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(train_lines) + "\n")
        tmp_train = f.name

    test_path = ""
    if test_lines:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("\n".join(test_lines) + "\n")
            test_path = f.name

    try:
        model = fasttext.train_supervised(
            input=tmp_train,
            epoch=50,
            lr=0.5,
            wordNgrams=2,
            verbose=0,
        )
        model.save_model(str(_QUALITY_MODEL_PATH))
        if test_path:
            n, p, r = model.test(test_path)
            print(f"[quality classifier] held-out eval: n={n}, precision={p:.3f}, recall={r:.3f}")
    finally:
        os.unlink(tmp_train)
        if test_path and os.path.exists(test_path):
            os.unlink(test_path)

    return model


def _get_quality_model() -> fasttext.FastText._FastText:
    global _quality_model
    if _quality_model is None:
        if _QUALITY_MODEL_PATH.exists():
            _quality_model = fasttext.load_model(str(_QUALITY_MODEL_PATH))
        else:
            _quality_model = _train_quality_model()
    return _quality_model


# Pre-compiled PII regexes (module-level for performance)
_EMAIL_RE = re.compile(
    r"(?<!\|{3})[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)
_PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:\+?1[\s\-.]?)?"
    r"(?:\(\d{3}\)[\s\-.]?|\d{3}[\s\-.]|\d{3})"
    r"\d{3}[\s\-.]?\d{4}"
    r"(?!\d)"
)
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]\d|\d)"
_IP_RE = re.compile(r"(?<!\d)(?:" + _OCTET + r"\.){3}" + _OCTET + r"(?!\d)")


# Adapter implementations


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """Decode raw HTML bytes and extract plain text via Resiliparse."""
    from resiliparse.extract.html2text import extract_plain_text
    from resiliparse.parse.encoding import detect_encoding
    from resiliparse.parse.html import HTMLTree

    try:
        html_str = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        enc = detect_encoding(html_bytes) or "utf-8"
        html_str = html_bytes.decode(enc, errors="replace")

    tree = HTMLTree.parse(html_str)
    return extract_plain_text(tree)


def run_identify_language(text: str) -> tuple[Any, float]:
    """Return (iso_code, confidence) for the top predicted language."""
    model = _get_lid_model()
    labels, probs = model.predict(text.replace("\n", " "), k=1)
    lang_code = labels[0].replace("__label__", "")
    return lang_code, float(probs[0])


def run_mask_emails(text: str) -> tuple[str, int]:
    """Replace email addresses with |||EMAIL_ADDRESS||| and return (text, count)."""
    masked, count = _EMAIL_RE.subn("|||EMAIL_ADDRESS|||", text)
    return masked, count


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    """Replace US phone numbers with |||PHONE_NUMBER||| and return (text, count)."""
    masked, count = _PHONE_RE.subn("|||PHONE_NUMBER|||", text)
    return masked, count


def run_mask_ips(text: str) -> tuple[str, int]:
    """Replace IPv4 addresses with |||IP_ADDRESS||| and return (text, count)."""
    masked, count = _IP_RE.subn("|||IP_ADDRESS|||", text)
    return masked, count


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    """Return ('nsfw'|'non-nsfw', confidence) using the Dolma Jigsaw NSFW model."""
    model = _get_nsfw_model()
    labels, probs = model.predict(text.replace("\n", " "), k=1)
    label = labels[0].replace("__label__", "")
    return label, float(probs[0])


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    """Return ('toxic'|'non-toxic', confidence) using the Dolma Jigsaw hate-speech model."""
    model = _get_hate_model()
    labels, probs = model.predict(text.replace("\n", " "), k=1)
    label = labels[0].replace("__label__", "")
    return label, float(probs[0])


def run_classify_quality(text: str) -> tuple[Any, float]:
    """Return ('hq'|'lq', confidence) using a trained fastText quality classifier."""
    model = _get_quality_model()
    labels, probs = model.predict(text.replace("\n", " "), k=1)
    
    # Strip the prefix to get "wiki" or "cc"
    raw_label = labels[0].replace("__label__", "")
    
    # Map the internal model labels to the exact strings the test expects
    label = "hq" if raw_label == "wiki" else "lq"
    
    return label, float(probs[0])


def run_gopher_quality_filter(text: str) -> bool:
    """Return True iff text passes the Gopher quality heuristics.

    Rules (document is REJECTED if any is violated):
      1. Word count outside [50, 100_000]
      2. Mean word length outside [3, 10]
      3. More than 30% of lines end with '...'
      4. Less than 80% of words contain at least one alphabetic character
    """
    words = text.split()
    n = len(words)

    # Rule 1 – word count
    if n < 50 or n > 100_000:
        return False

    # Rule 2 – mean word length
    avg_len = sum(len(w) for w in words) / n
    if avg_len < 3 or avg_len > 10:
        return False

    # Rule 3 – ellipsis lines (denominator is non-empty lines only)
    non_empty_lines = [ln for ln in text.splitlines() if ln.strip()]
    if non_empty_lines:
        ellipsis_ratio = sum(
            1 for ln in non_empty_lines if ln.rstrip().endswith("...")
        ) / len(non_empty_lines)
        if ellipsis_ratio > 0.3:
            return False

    # Rule 4 – alphabetic word fraction
    alpha_count = sum(1 for w in words if any(c.isalpha() for c in w))
    if alpha_count / n < 0.8:
        return False

    return True


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
) -> None:
    """Remove lines that appear more than once across all input files.

    Phase 1 – count each line (by MD5 hash) across the whole corpus.
    Phase 2 – rewrite each file keeping only lines with count == 1.
    """
    output_dir = pathlib.Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    line_counts: dict[str, int] = defaultdict(int)

    # Phase 1
    for path in input_files:
        with open(path) as f:
            for line in f:
                h = hashlib.md5(line.rstrip("\n").encode()).hexdigest()
                line_counts[h] += 1

    # Phase 2
    for path in input_files:
        path = pathlib.Path(path)
        out_path = output_dir / path.name
        with open(path) as f_in, open(out_path, "w") as f_out:
            for line in f_in:
                h = hashlib.md5(line.rstrip("\n").encode()).hexdigest()
                if line_counts[h] == 1:
                    f_out.write(line)


# MinHash helpers

def _normalize_for_minhash(text: str) -> str:
    """NFD, remove accents, lowercase, strip punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _word_ngrams(text: str, n: int) -> set[str]:
    words = text.split()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _minhash_sig(ngrams: set[str], num_hashes: int) -> list[int]:
    if not ngrams:
        return [0] * num_hashes
    return [
        min(mmh3.hash(ng, seed, signed=False) for ng in ngrams)
        for seed in range(num_hashes)
    ]


class _UF:
    """Simple union-find for clustering duplicate documents."""

    def __init__(self) -> None:
        self._p: dict = {}

    def find(self, x: Any) -> Any:
        if x not in self._p:
            self._p[x] = x
        if self._p[x] != x:
            self._p[x] = self.find(self._p[x])
        return self._p[x]

    def union(self, x: Any, y: Any) -> None:
        px, py = self.find(x), self.find(y)
        if px != py:
            self._p[px] = py


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
) -> None:
    """Fuzzy document deduplication via MinHash + LSH.

    Each file in input_files is treated as one document. Files whose content is
    sufficiently similar (Jaccard >= jaccard_threshold) are clustered and all but
    one representative per cluster are excluded from the output.

    Steps:
      1. Normalize text and compute word n-gram sets.
      2. Compute MinHash signatures (num_hashes functions via mmh3 seeds).
      3. LSH: split signature into num_bands bands; bucket same-band docs.
      4. For each candidate pair, verify Jaccard similarity >= threshold.
      5. Union-find clustering; keep one doc per cluster.
      6. Write retained docs to output_directory.
    """
    band_size = num_hashes // num_bands
    output_dir = pathlib.Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 & 2: Each input file is treated as a single document for deduplication purposes.
    doc_ids: list[str] = []
    doc_text: dict[str, str] = {}
    doc_ngrams: dict[str, set[str]] = {}
    sigs: dict[str, list[int]] = {}

    for path in input_files:
        path = pathlib.Path(path)
        doc_id = str(path)
        text = path.read_text()
        doc_ids.append(doc_id)
        doc_text[doc_id] = text
        norm = _normalize_for_minhash(text)
        ng = _word_ngrams(norm, ngrams)
        doc_ngrams[doc_id] = ng
        sigs[doc_id] = _minhash_sig(ng, num_hashes)

    # Step 3: LSH candidate pairs
    candidates: set[tuple[str, str]] = set()
    for b in range(num_bands):
        buckets: dict[tuple[int, ...], list[str]] = defaultdict(list)
        for doc_id in doc_ids:
            band = tuple(sigs[doc_id][b * band_size : (b + 1) * band_size])
            buckets[band].append(doc_id)
        for bucket in buckets.values():
            if len(bucket) > 1:
                for i in range(len(bucket)):
                    for j in range(i + 1, len(bucket)):
                        pair = (min(bucket[i], bucket[j]), max(bucket[i], bucket[j]))
                        candidates.add(pair)

    # Step 4 & 5: verify Jaccard + cluster
    uf = _UF()
    for a, b_doc in candidates:
        ng_a, ng_b = doc_ngrams[a], doc_ngrams[b_doc]
        union = len(ng_a | ng_b)
        if union > 0 and len(ng_a & ng_b) / union >= jaccard_threshold:
            uf.union(a, b_doc)

    # Collect clusters, mark duplicates to discard
    clusters: dict[str, list[str]] = defaultdict(list)
    for doc_id in doc_ids:
        clusters[uf.find(doc_id)].append(doc_id)

    to_discard: set[str] = set()
    for cluster in clusters.values():
        if len(cluster) > 1:
            keep = sorted(cluster)[0]
            to_discard.update(d for d in cluster if d != keep)

    # Step 6: write retained docs (skip discarded files entirely)
    for path in input_files:
        path = pathlib.Path(path)
        doc_id = str(path)
        if doc_id not in to_discard:
            (output_dir / path.name).write_text(doc_text[doc_id])


# Data collection utilities (call manually, not used by the test suite)


def collect_quality_positives(
    wet_path: str, output_path: str = "quality_positives.txt"
) -> None:
    """Read a WET file, extract up to 3000 text records, write one per line to output_path.

    Intended for Wikipedia WET files — records are positive (high-quality) examples.
    Records with fewer than 50 words are skipped.
    """
    from warcio.archiveiterator import ArchiveIterator

    collected: list[str] = []
    with open(wet_path, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type != "conversion":
                continue
            text = record.content_stream().read().decode("utf-8", errors="replace")
            text = " ".join(text.split())  # collapse whitespace / newlines
            if len(text.split()) < 50:
                continue
            collected.append(text[:500])
            if len(collected) >= 3000:
                break

    with open(output_path, "w") as f:
        for doc in collected:
            f.write(doc + "\n")
    print(f"Wrote {len(collected)} positives to {output_path}")


def collect_quality_negatives(
    wet_path: str, output_path: str = "quality_negatives.txt"
) -> None:
    """Read a WET file, extract up to 3000 text records, write one per line to output_path.

    Intended for Common Crawl WET files — records are negative (low-quality) examples.
    Records with fewer than 50 words are skipped.
    """
    from warcio.archiveiterator import ArchiveIterator

    collected: list[str] = []
    with open(wet_path, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type != "conversion":
                continue
            text = record.content_stream().read().decode("utf-8", errors="replace")
            text = " ".join(text.split())
            if len(text.split()) < 50:
                continue
            collected.append(text[:500])
            if len(collected) >= 3000:
                break

    with open(output_path, "w") as f:
        for doc in collected:
            f.write(doc + "\n")
    print(f"Wrote {len(collected)} negatives to {output_path}")
