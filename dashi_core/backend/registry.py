from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable

from .accelerated import AcceleratedBackend
from .base import Backend
from .cpu import CPUBackend


_registry: Dict[str, Backend] = {
    "cpu": CPUBackend(),
    "accelerated": AcceleratedBackend(),
}
_active_backend: Backend = _registry["cpu"]


def register_backend(name: str, backend: Backend) -> None:
    _registry[name] = backend


def set_backend(name: str) -> None:
    if name not in _registry:
        raise ValueError(f"Unknown backend: {name}")
    global _active_backend
    _active_backend = _registry[name]


def get_backend() -> Backend:
    return _active_backend


@contextmanager
def use_backend(name: str):
    previous = get_backend()
    set_backend(name)
    try:
        yield get_backend()
    finally:
        # Restore prior backend
        global _active_backend
        _active_backend = previous


def list_backends() -> Iterable[str]:
    return tuple(_registry.keys())
