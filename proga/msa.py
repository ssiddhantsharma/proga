import hashlib
import os
import shutil

from protenix.utils.logger import get_logger
from runner.msa_search import msa_search

logger = get_logger(__name__)


def seq_hash(seq: str) -> str:
    return hashlib.sha256(seq.encode("utf-8")).hexdigest()[:16]


class MSACache:
    def __init__(self, root_dir: str, mode: str = "protenix"):
        self.root = os.path.expanduser(root_dir)
        self.mode = mode
        os.makedirs(self.root, exist_ok=True)

    def _dir(self, seq: str) -> str:
        return os.path.join(self.root, seq_hash(seq))

    def paths(self, seq: str) -> dict[str, str]:
        d = self._dir(seq)
        out: dict[str, str] = {}
        paired, unpaired = os.path.join(d, "pairing.a3m"), os.path.join(d, "non_pairing.a3m")
        if os.path.exists(paired):
            out["pairedMsaPath"] = paired
        if os.path.exists(unpaired):
            out["unpairedMsaPath"] = unpaired
        return out

    def is_cached(self, seq: str) -> bool:
        return bool(self.paths(seq))

    def get_or_compute(self, seqs: list[str]) -> dict[str, dict[str, str]]:
        unique = list(dict.fromkeys(seqs))  # preserve order, dedup
        missing = [s for s in unique if not self.is_cached(s)]
        if missing:
            self._search_and_cache(missing)
        return {s: self.paths(s) for s in unique}

    def _search_and_cache(self, missing: list[str]) -> None:

        tmp_dir = os.path.join(self.root, "_search_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        logger.info("Running MSA search for %d sequence(s) [mode=%s]", len(missing), self.mode)
        # msa_search returns one result subdir per input sequence, in order.
        subdirs = msa_search(missing, tmp_dir, mode=self.mode)
        if len(subdirs) != len(missing):
            raise RuntimeError(
                f"MSA search returned {len(subdirs)} subdirs for {len(missing)} sequences."
            )
        for seq, subdir in zip(missing, subdirs):
            dest = self._dir(seq)
            os.makedirs(dest, exist_ok=True)
            for fname in ("pairing.a3m", "non_pairing.a3m"):
                src = os.path.join(subdir, fname)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(dest, fname))
            if not self.is_cached(seq):
                logger.warning("No MSA produced for sequence %s (len %d).",
                               seq_hash(seq), len(seq))
        shutil.rmtree(tmp_dir, ignore_errors=True)
