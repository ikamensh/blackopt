import signal

class UserTimeoutError(BaseException):
    def __str__(self):
        return "Timed Out"

def timeout(t):
    def decorate(f):
        def handler(signum, frame):
            print(f"Function `{f.__name__}` timed out.")
            raise UserTimeoutError()
        def new_f(*args, **kwargs):
            old = signal.signal(signal.SIGALRM, handler)
            assert old is signal.SIG_DFL
            signal.setitimer(signal.ITIMER_REAL, t)
            try:
                return f(*args, **kwargs)
            finally:
                try:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old)
                except UserTimeoutError:
                    signal.signal(signal.SIGALRM, old)
        return new_f
    return decorate

import time



if __name__ == '__main__':

    @timeout(3.3)
    def mytest():
        print("Start")
        try:
            for i in range(1, 10):
                time.sleep(1)
                print("%d seconds have passed" % i)
        except Exception:
            print("Caught something")

    mytest()