from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .defects import DEFECTS


def _doc(d: dict) -> str:
    marker = d["marker"][2] if d.get("marker") else ""
    return " ".join([d["title"], d["description"], d["root_cause"], marker])


class DefectIndex:
    def __init__(self, defects: list[dict] | None = None):
        self.defects = defects or DEFECTS
        self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True, token_pattern=r"[A-Za-z0-9_#]{2,}")
        self._mat = self._vec.fit_transform([_doc(d) for d in self.defects])

    def search(self, query: str, k: int = 3) -> list[dict]:
        if not query.strip():
            return []
        sims = cosine_similarity(self._vec.transform([query]), self._mat)[0]
        order = sims.argsort()[::-1][:k]
        return [dict(id=self.defects[i]["id"], title=self.defects[i]["title"], fault_class=self.defects[i]["fault_class"],
                     root_cause=self.defects[i]["root_cause"], fix_hint=self.defects[i]["fix_hint"],
                     score=round(float(sims[i]), 4)) for i in order]
