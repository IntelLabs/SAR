
import multiprocessing as mp
import traceback
import pytest

def handle_mp_exception(mp_dict):
    msg = mp_dict.get('traceback', "")
    for e_arg in mp_dict['exception'].args:
        msg += str(e_arg)
    pytest.fail(str(msg), pytrace=False)
    
def sar_test(func):
    """
    A decorator function that wraps all SAR tests with the primary objective
    of facilitating module imports in tests without affecting other tests.

    :param func: The function that serves as the entry point to the test.
    :type func: function
    :returns: A function that encapsulates the pytest function.
    """
    def test_wrapper(*args, **kwargs):
        """
        The wrapping process involves defining another nested function, which is then invoked by a newly spawned process.
        function spawns a new process and uses the "join" method to wait for the results.
        Upon completion of the process, error and result handling are performed.
        """
        def process_wrapper(func, mp_dict, *args, **kwargs):
            try:
                result = func(*args, **kwargs)
                mp_dict["result"] = result
            except Exception as e:
                mp_dict['traceback'] = str(traceback.format_exc())
                mp_dict["exception"] = e

        manager = mp.Manager()
        mp_dict = manager.dict()

        mp_args = (func, mp_dict) + args
        p = mp.Process(target=process_wrapper, args=mp_args, **kwargs)
        p.start()
        p.join()

        if 'exception' in mp_dict:
            handle_mp_exception(mp_dict)

        return mp_dict["result"]
    return test_wrapper
