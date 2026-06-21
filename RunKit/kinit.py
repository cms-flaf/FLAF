import shutil
import threading

from .run_tools import ps_call

cond = threading.Condition()


def update_kinit(verbose=0):
    if shutil.which("kinit"):
        ps_call(["kinit", "-R"], expected_return_codes=None, verbose=verbose)
    if shutil.which("aklog"):
        ps_call(["aklog"], expected_return_codes=None, verbose=verbose)


def update_kinit_thread():
    timeout = 60.0 * 60  # 1 hour
    cond.acquire()
    while not cond.wait(timeout):
        update_kinit(verbose=1)
    cond.release()
