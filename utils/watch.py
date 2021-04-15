from collections import OrderedDict
import logging
import time

log = logging.getLogger()


class Watch:
    def __init__(self, name, logger='main', start_now=False):
        self.name = name
        self.logger = logging.getLogger(logger)
        if start_now:
            self.start = time.time()
        else:
            self.start = None

    def __enter__(self):
        self.go()

    def __exit__(self, type, value, trace_back):
        self.stop()

    def go(self):
        self.start = time.time()

    def stop(self, tag="", silent=False):
        secs = time.time() - self.start
        name = self.name + (f"_{tag}" if tag else "")
        out = _secs_to_dhms_str(secs, delimiter=" ")
        if not silent:
            self.logger.info(f'Watch({name}): {out}')
        return out

    def touch(self, tag="", silent=False):
        if self.start is not None:
            out = self.stop(tag=tag, silent=silent)
        else:
            out = None
        return out

    def touch_and_go(self, tag="", silent=False):
        out = self.touch(tag, silent)
        self.go()
        return out


class TimeOfArrivalEstimator:
    """A very simple ETA predictor. In the early stages of training, 
    estimates may be inaccurate and fluctuate along the steps, especially if 
    .esitimate_step*() is called irregularly due to the intermittent validation. 
    Nevertheless, it seems useful enough as the number of steps increases.
    """
    def __init__(self, total_step):
        self._total_step = total_step
        self._cur_step = 0
        self._start = None
        
    @classmethod
    def init_from_epoch_steps(cls, epochs, epoch_steps):
        return cls(epochs * epoch_steps)     
    
    def estimate_step(self):
        if self._start is None:
            self._start = time.time()
            return -1
        self._cur_step += 1
        secs = time.time() - self._start
        remained_step = self._total_step - self._cur_step
        return secs / self._cur_step * remained_step
    
    def estimate_step_str(self):
        secs = self.estimate_step()
        if secs == -1:
            return "N/A"
        return _secs_to_dhms_str(secs, first_n=2)


def _secs_to_dhms(secs, keys=('d', 'h', 'm', 's')):
    return OrderedDict({
        keys[0]: int((secs // (3600 * 24))),
        keys[1]: int((secs %  (3600 * 24)) // 3600),
        keys[2]: int((secs %  (3600 * 24)) %  3600 // 60),
        keys[3]: int((secs %  (3600 * 24)) %  3600 %  60),
    })

def _secs_to_dhms_str(secs, first_n=-1, delimiter=""):
    """return DHMS formatted string that contains first K non-zero elements.
        if first_n= 2, for example:
        >> _secs_to_dhms_str(12)        -> '00m 12s'
        >> _secs_to_dhms_str(400)       -> '06m 40s'
        >> _secs_to_dhms_str(12000)     -> '03h 20m'
        >> _secs_to_dhms_str(25 * 3600) -> '01d 01h'
    """
    assert first_n <= 4
    dhms = _secs_to_dhms(secs)
    dhms_values = [v for v in dhms.values()]
    
    # first non-zero id
    if first_n > 0:
        idx = next((i for i, x in enumerate(dhms_values) if x), 5)
        idx = max(0, idx + min(0, 4 - (idx + first_n)))
        dhms = {k: v for i, (k, v) in enumerate(dhms.items()) 
                if idx <= i < idx + first_n}
    return delimiter.join([f"{v:02d}" + k for k, v in dhms.items()
                           if type(v) == int and v >= 0])
