import pytest

from logtriage.generator import generate_dataset
from logtriage.signature import TemplateStats


@pytest.fixture(scope="session")
def dataset():
    return generate_dataset(120, seed=11, hard=False)


@pytest.fixture(scope="session")
def stats():
    base = generate_dataset(150, seed=3, hard=False, prefix="b")
    return TemplateStats().fit([r["text"] for r in base if r["label"] == "PASS"])


class ScriptedLLM:
    name = "scripted"

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0

    def complete(self, system, user):
        self.calls += 1
        return self.answers[min(self.calls - 1, len(self.answers) - 1)]
