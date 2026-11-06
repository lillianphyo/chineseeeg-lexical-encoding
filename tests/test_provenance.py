import subprocess

from zuco_llm_align import provenance


def test_git_hash_returns_head(monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "abc123\n")
    assert provenance.git_hash() == "abc123"


def test_git_hash_outside_a_repository_is_unknown(monkeypatch):
    def fail(*a, **k):
        raise subprocess.CalledProcessError(128, "git")
    monkeypatch.setattr(subprocess, "check_output", fail)
    assert provenance.git_hash() == "unknown"
