from contextlib import ExitStack
import io
import logging
import os
import sys
import types
from utils.color import Colorer, ColorerContext, decolorize
C = Colorer.instance()


def formatter_kwargs():
    return dict(
        # fmt="%(asctime)s [%(filename)s:%(lineno)d] %(msg)s",
        datefmt='%d/%m/%y %H:%M'
    )


def get_log_level(level):
    return getattr(logging, level.upper())


def set_stream_handler(name, level):
    assert isinstance(level, str)

    # set level
    logger = logging.getLogger(name)
    logger.setLevel(get_log_level(level))

    # stdout stream handler
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    formatter = MyFormatter(**formatter_kwargs())
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(get_log_level(level))
    logger.stream_handler = stream_handler
    logger.addHandler(stream_handler)

    # stdio newline handler
    newline_handler = logging.StreamHandler(stream=sys.stdout)
    newline_handler.setFormatter(fmt='')
    newline_handler.setLevel(get_log_level(level))

    # newline() to switch the newline_handler
    def newline(self, num_lines=1, log_level='info'):
        # handler swithching trick
        self.removeHandler(self.stream_handler)
        self.addHandler(self.newline_handler)

        for _ in range(num_lines):
            getattr(self, log_level)('')

        self.removeHandler(self.newline_handler)
        self.addHandler(self.stream_handler)

    logger.newline_handler = newline_handler
    logger.newline = types.MethodType(newline, logger)

    return logger


def set_file_handler(name, level, save_dir, filename):
    assert isinstance(level, str)

    # set level
    logger = logging.getLogger(name)
    logger.setLevel(get_log_level(level))

    # file handler
    log_file = os.path.join(save_dir, filename)
    file_handler = logging.FileHandler(log_file)
    formatter = MyFormatter(no_color=True, **formatter_kwargs())
    file_handler.setFormatter(formatter)
    file_handler.setLevel(get_log_level(level))
    logger.addHandler(file_handler)

    return logger
    

class MyFormatter(logging.Formatter):
    def __init__(self, no_color=False, *args, **kwargs):
        self.no_color = no_color
        super().__init__(*args, **kwargs)
        
    def format(self, record):
        level_name = None   
        
        

        if record.levelno == logging.INFO:
            level_name = C.faint("[INFO]")
        elif record.levelno == logging.DEBUG:
            level_name = C.debug("[DEBUG]")
        elif record.levelno == logging.WARNING:
            level_name = C.warning("[WARN]")
        elif record.levelno == logging.ERROR:
            level_name = C.error("[ERROR]")
        elif record.levelno == logging.CRITICAL:
            level_name = C.error("[CRITICAL]")

        if level_name:
            format_orig = self._style._fmt
            self._style._fmt = (
                f"{C.faint('%(asctime)s')} {level_name} "
                f"{C.faint('[%(module)s:%(lineno)d]')} %(msg)s"
            )

        result = logging.Formatter.format(self, record)
        if self.no_color:
            result = decolorize(result)

        if level_name:
            self._style._fmt = format_orig

        return result
