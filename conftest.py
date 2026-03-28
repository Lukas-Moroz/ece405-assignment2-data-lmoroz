import shutil
import pathlib

# Wipe the cs336-data/cs336-basics tests pycache before every session so that
# the editable-installed packages never win the "tests" module namespace race.
def pytest_configure(config):
    for subdir in ("cs336-data", "cs336-basics"):
        cache = pathlib.Path(__file__).parent / subdir / "tests" / "__pycache__"
        if cache.exists():
            shutil.rmtree(cache)

collect_ignore_glob = ["cs336-data/*", "cs336-basics/*"]
