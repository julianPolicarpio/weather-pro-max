"""Design-freeze fingerprint (spec §3.8): one hash over every file that shapes a stored prediction or
score, so a dataset, a walk-forward run and a report can each be tied to the code that produced them."""
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FINGERPRINT_FILES = ("src/citycast/asof.py", "src/citycast/features.py", "src/citycast/dataset.py",
                     "src/citycast/qmodel.py", "src/citycast/metrics.py", "src/citycast/walkforward.py",
                     "src/citycast/evaluate.py", "src/citycast/fingerprint.py", "src/citycast/stations.py",
                     "src/highcast/latentmax.py", "src/highcast/buckets.py", "src/highcast/timeutil.py",
                     "config/citycast_stations.yaml")


def code_fingerprint() -> str:
    """sha256 over each path (relative to the repo root) followed by its bytes, in the order above."""
    h = hashlib.sha256()
    for rel in FINGERPRINT_FILES:
        h.update(rel.encode())
        h.update((REPO_ROOT / rel).read_bytes())
    return h.hexdigest()
