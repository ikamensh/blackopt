import signal

class UserTimeoutError(Exception):
    def __str__(self):
        return "Timed Out"

def timeout(t):
    def decorate(f):
        def handler(signum, frame):
            raise UserTimeoutError()
        def new_f(*args, **kwargs):
            old = signal.signal(signal.SIGALRM, handler)
            # assert old is signal.SIG_DFL
            signal.setitimer(signal.ITIMER_REAL, t)
            try:
                result = f(*args, **kwargs)
            finally:
                # reinstall the old signal handler
                signal.signal(signal.SIGALRM, old)
                # cancel the alarm
                signal.setitimer(signal.ITIMER_REAL, 0)
            return result
        return new_f
    return decorate

import time



if __name__ == '__main__':

    @timeout(3.3)
    def mytest():
        print("Start")
        for i in range(1, 10):
            time.sleep(1)
            print("%d seconds have passed" % i)

    mytest()