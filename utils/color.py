import re


def decolorize(text):
    # 7-bit C1 ANSI sequences
    ansi_escape = re.compile(r'''
        \x1B  # ESC
        (?:   # 7-bit C1 Fe (except CSI)
            [@-Z\\-_]
        |     # or [ for CSI, followed by a control sequence
            \[
            [0-?]*  # Parameter bytes
            [ -/]*  # Intermediate bytes
            [@-~]   # Final byte
        )
    ''', re.VERBOSE)
    return ansi_escape.sub('', text)
    

class SingletonInstance:
    _instance = None

    @classmethod
    def _getInstance(cls):
        return cls._instance

    @classmethod
    def instance(cls, *args, **kargs):
        cls._instance = cls(*args, **kargs)
        cls.instance = cls._getInstance
        return cls._instance
    
    
class Colorer(SingletonInstance):
    """A class that helps to colorize text. User can choose whether he/she will 
    use the functionality anywhere in the code, in a global manner.
    """
    def __init__(self):
        self._enabled = True
        
    @property
    def enabled(self):
        return self._enabled
                
    def set_enabled(self, enabled):
        self._enabled = enabled
        
    def __getattr__(self, name):
        if name == '_colorize':
            return self.__getattribute__(name) 
        elif name.startswith('C_'):
            if not self._enabled:
                return ""
            return getattr(Pallete, name)
        else:
            def _colorize(msg):
                return self._colorize(msg, name)
            return _colorize
            
    def _colorize(self, msg: str, color: str):
        c_color = self.__getattr__(f"C_{color.upper()}")
        return f"{c_color}{msg}{self.C_END}"
    

class ColorerContext(object):
    def __init__(self, colorer, enabled):
        self._colorer = colorer
        self._enabled = enabled
        
    def __enter__(self):
        self._enabled_prev = self._colorer._enabled
        self._colorer.set_enabled(self._enabled)
        return self
    
    def __exit__(self, type, value, trace_back): 
        self._colorer.set_enabled(self._enabled_prev)


class Pallete(object):
    C_END = "\33[0m"
    C_BOLD = "\33[1m"
    C_FAINT = "\33[2m"
    C_ITALIC = "\33[3m"
    C_UNDERLINE = '\33[4m'
    C_BLINK = "\33[5m"
    C_BLINK2 = "\33[6m"
    C_SELECTED = "\33[7m"

    C_BLACK = "\33[30m"
    C_RED = "\33[31m"
    C_GREEN = "\33[32m"
    C_YELLOW = "\33[33m"
    C_BLUE = "\33[34m"
    C_VIOLET = "\33[35m"
    C_CYAN = "\33[36m"
    C_WHITE = "\33[37m"

    C_BLACKBG = "\33[40m"
    C_REDBG = "\33[41m"
    C_GREENBG = "\33[42m"
    C_YELLOWBG = "\33[43m"
    C_BLUEBG = "\33[44m"
    C_VIOLETBG = "\33[45m"
    C_CYANBG = "\33[46m"
    C_WHITEBG = "\33[47m"

    C_GREY = "\33[90m"
    C_RED2 = "\33[91m"
    C_GREEN2 = "\33[92m"
    C_YELLOW2 = "\33[93m"
    C_BLUE2 = "\33[94m"
    C_VIOLET2 = "\33[95m"
    C_CYAN2 = "\33[96m"
    C_WHITE2 = "\33[97m"

    C_GREYBG = "\33[100m"
    C_REDBG2 = "\33[101m"
    C_GREENBG2 = "\33[102m"
    C_YELLOWBG2 = "\33[103m"
    C_BLUEBG2 = "\33[104m"
    C_VIOLETBG2 = "\33[105m"
    C_CYANBG2 = "\33[106m"
    C_WHITEBG2 = "\33[107m"
    
    C_DEBUG = '\033[5;95m'
    C_WARNING = '\033[5;93m'
    C_ERROR = '\033[5;91m'
