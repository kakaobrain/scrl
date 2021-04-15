import re
from tqdm import tqdm
import logging

from distributed import comm
from utils import Colorer

LINE_DISPLAY_FREQ = 0.1
log = logging.getLogger('main')
C = Colorer.instance()


class DistributedProgressDisplayer(object):
    """Progress logger for the distributed learning environment. Tqdm progress 
    bars are only used for local main processes. File logging is done in the
    every non-main processes where the status is stored less often to prevent 
    the log file from getting too large.
    """
    def __init__(self, max_steps, no_pbar=False, local_disp=True, desc=""):
        self.step = 0
        self.max_steps = max_steps
        self.no_pbar = no_pbar
        self.local_disp = local_disp
        self.desc = desc
        
        if not self.no_pbar and self.is_main_process():
            bar_format = (
                '{desc}{percentage:3.0f}%|'
                f"{C.green('{bar:10}')}"
                '| it:{n_fmt}/{total_fmt}{postfix} [{rate_fmt}]'
            )
            self.pbar = tqdm(range(max_steps), bar_format=bar_format)
            self.pbar.set_description(self.desc)
        else:
            line_steps = round(max_steps * LINE_DISPLAY_FREQ)
            self.disp_steps = max(1, line_steps)
            
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, trace_back): 
        self.close()
        
    def is_main_process(self):
        if self.local_disp:
            return comm.is_local_main_process()
        else:
            return comm.is_main_process()
        
    def update_with_postfix(self, postfix):
        self.step += 1
        if not self.no_pbar and self.is_main_process():
            self.pbar.set_postfix_str(postfix)
            self.pbar.update()
        else:
            if self.step % self.disp_steps == 0:
                msg = self.desc + " " + postfix
                log.info(msg + f" (step: {self.step}/{self.max_steps})")
        
    def close(self):
        if not self.no_pbar and self.is_main_process():
            self.pbar.close()
