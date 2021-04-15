"""Reference: https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/"""
from contextlib import contextmanager
import ctypes
import io
import logging
import os
import sys
import tempfile
from typing import Any
import warnings

import torch

log = logging.getLogger('main')
libc = ctypes.CDLL(None)
c_stderr = ctypes.c_void_p.in_dll(libc, 'stderr')


class detect_anomaly_with_redirected_cpp_error(object):
    """During anomaly detection using PyTorch API, some of the exception 
    messages are outputted from C++ level so that those cannot be captured by 
    python loggers. This class helps to redirect those messages.
    """
    def __init__(self, mode: bool) -> None:
        self.prev = torch.is_anomaly_enabled()
        self.mode = mode
        torch.set_anomaly_enabled(mode)
        if self.mode:
            warnings.warn('Anomaly Detection has been enabled. '
                          'This mode will increase the runtime '
                          'and should only be enabled for debugging.', stacklevel=2)
            self.stream = io.BytesIO()
            # The original fd stderr points to. Usually 2 on POSIX systems.
            self.stderr_fd_origin = sys.stderr.fileno()
            # Make a copy of the original stderr fd in stderr_fd_copy
            self.stderr_fd_copy = os.dup(self.stderr_fd_origin)
            # Create a temporary file and redirect stderr to it
            self.tfile = tempfile.TemporaryFile(mode='w+b')
            self._redirect_stderr(self.tfile.fileno())

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type, exc_val, traceback) -> None:
        torch.set_anomaly_enabled(self.prev)
        if self.mode:
            # redirect stderr back to the saved fd
            self._redirect_stderr(self.stderr_fd_copy)
            # Copy contents of temporary file to the given stream
            self.tfile.flush()
            self.tfile.seek(0, io.SEEK_SET)
            self.stream.write(self.tfile.read())
            self.tfile.close()
            os.close(self.stderr_fd_copy)
            stream_value = self.stream.getvalue().decode('utf-8')
            
            if exc_val:
                raise RuntimeError(stream_value).with_traceback(traceback)
            else:
                log.warning(stream_value)
            
    def _redirect_stderr(self, to_fd):
        """Redirect stderr to the given file descriptor."""
        # Flush the C-level buffer stderr
        libc.fflush(c_stderr)
        # Flush and close sys.stderr - also closes the file descriptor (fd)
        sys.stderr.flush()
        # Make stderr_fd_origin point to the same file as to_fd
        os.dup2(to_fd, self.stderr_fd_origin)
