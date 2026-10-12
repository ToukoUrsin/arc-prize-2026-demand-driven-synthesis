"""Python-level scientific inference controls; not an OS sandbox. MIT-0."""
from collections.abc import Mapping
from contextlib import contextmanager
import builtins
import io
import os
import socket
import sys


class Guard(Mapping):
    """Keep only allowed payloads so mapping copies cannot reveal labels."""
    def __init__(self, value, allowed):
        self._allowed = frozenset(allowed)
        self._data = {k: value[k] for k in self._allowed if k in value}

    def __getitem__(self, key):
        if key not in self._allowed:
            raise AssertionError('Forbidden inference field: ' + str(key))
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


_ACTIVE = False


def _audit_event(event, args):
    if _ACTIVE and (event == 'open' or
                   event.startswith(('socket.', 'subprocess.', 'ctypes.dlopen', 'ctypes.dlsym')) or
                   event in {'os.system', 'os.posix_spawn', 'os.exec', 'os.fork', 'os.forkpty',
                             'os.startfile', 'os.listdir', 'os.scandir'}):
        raise AssertionError('Forbidden inference audit event: ' + event)


sys.addaudithook(_audit_event)


@contextmanager
def inference_access():
    global _ACTIVE
    assert not _ACTIVE, 'Nested inference guard'
    targets = [(builtins, 'open'), (io, 'open'), (os, 'open'), (socket, 'socket'),
               (socket, 'getaddrinfo'), (socket, 'create_connection')]
    originals = [(module, name, getattr(module, name)) for module, name in targets]
    def denied(*args, **kwargs):
        raise AssertionError('Forbidden inference file/network access')
    try:
        _ACTIVE = True
        for module, name in targets:
            setattr(module, name, denied)
        yield
    finally:
        _ACTIVE = False
        for module, name, original in originals:
            setattr(module, name, original)


def blind_view(task):
    return Guard({'train': [Guard(p, {'input', 'output'}) for p in task['train']],
                  'test': [Guard(p, {'input'}) for p in task['test']]}, {'train', 'test'})
