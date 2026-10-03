"""Content-addressed stage results. No credentials are persisted.

The caller holds the project lock; independent async stages have separate manifests.
Only a completed manifest with matching file hashes may be reused.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class Project:
    def __init__(self, root: Path, name: str):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', name):
            raise ValueError('Project name: use 1–80 letters A-Z, digits, _ or -')
        self.path = root / name
        self.path.mkdir(parents=True, exist_ok=True)

    def stage(self, name: str, index: int, inputs: dict) -> 'Stage':
        return Stage(self.path / name / f'{index:03d}' / fingerprint(inputs), inputs)


class Stage:
    def __init__(self, path: Path, inputs: dict):
        self.path = path
        self.inputs = inputs
        self.manifest = path / 'state.json'

    def cached(self) -> dict | None:
        try:
            data = json.loads(self.manifest.read_text(encoding='utf-8'))
            if data.get('status') != 'completed' or data.get('inputs') != self.inputs:
                return None
            for name, expected in data['files'].items():
                path = self.path / name
                if not path.is_file() or path.stat().st_size == 0 or file_hash(path) != expected:
                    return None
            return data['result']
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def start(self) -> None:
        atomic_json(self.manifest, {'status': 'running', 'inputs': self.inputs})

    def finish(self, result: dict, files=()) -> None:
        hashes = {}
        for path in files:
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f'Missing or empty output: {path.name}')
            hashes[str(path.relative_to(self.path))] = file_hash(path)
        atomic_json(self.manifest, {'status': 'completed', 'inputs': self.inputs,
                                   'result': result, 'files': hashes})

    def fail(self, reason: str) -> None:
        atomic_json(self.manifest, {'status': 'failed', 'inputs': self.inputs, 'error': reason})
