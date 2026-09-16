"""Opt-in cache locking for the CUHK research filesystem, where flock hangs."""
import os


def configure_cache_locks():
    if os.environ.get('FYP_SOFT_FILELOCK') != '1':
        return
    import filelock

    class ResearchFileLock(filelock.SoftFileLock):
        def __init__(self, lock_file, *args, **kwargs):
            # Native file locks leave their empty files behind after release.
            # Keep soft locks separate so those files do not block acquisition.
            super().__init__(str(lock_file) + '.soft', *args, **kwargs)

    # Configure before importing datasets or huggingface_hub, which bind this
    # class on import. All writers to this cache must use the same lock mode.
    filelock.FileLock = ResearchFileLock
    print('Using file-based cache locks for CUHK research storage.', flush=True)
