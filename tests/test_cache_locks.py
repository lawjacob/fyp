"""Exercise the real Hugging Face and Datasets lock types in a fresh process."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class CacheLockTests(unittest.TestCase):
    def test_cache_locks_exclude_other_writers_and_release(self):
        code = '''
import tempfile
from pathlib import Path
from benchmarks.cache_locks import configure_cache_locks
configure_cache_locks()
from filelock import SoftFileLock, Timeout
from huggingface_hub.utils._fixes import FileLock as HubLock
from datasets.utils.filelock import FileLock as DatasetLock
for lock_type in (HubLock, DatasetLock):
    assert issubclass(lock_type, SoftFileLock)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'cache.lock'
        path.touch()  # A leftover native lock must not block the soft lock.
        first, second = lock_type(path), lock_type(path)
        with first:
            try:
                second.acquire(timeout=0)
            except Timeout:
                pass
            else:
                raise AssertionError('A second writer acquired the lock')
        with second.acquire(timeout=0):
            assert Path(str(path) + '.soft').exists()
        assert not Path(str(path) + '.soft').exists()
'''
        subprocess.run([sys.executable, '-c', code], check=True, timeout=60,
                       cwd=Path(__file__).resolve().parents[1],
                       env={**os.environ, 'FYP_SOFT_FILELOCK': '1'})
