import os
import sys


class Logger():


    def __init__(self):
        self.fp = None
        self.logfile = "log.txt"


    def enable(self):
        # redirect output to log file
        open(self.logfile, "a").close()
        self.fp = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(self.logfile, os.O_WRONLY)


    def disable(self):
        os.close(1)
        os.dup(self.fp)
        os.close(self.fp)
        
    def set_log_name(self,logfile):
        self.logfile = logfile