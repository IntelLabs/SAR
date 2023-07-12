import multiprocessing as mp
import traceback
import pytest
import functools
import tempfile


def handle_mp_exception(mp_dict):
    """
    Used to handle exceptions that occurred in child processes
    
    :param mp_dict: Dictionary that is shared between different processes
    :type mp_dict: multiprocessing.managers.DictProxy
    """
    msg = mp_dict.get('traceback', "")
    for e_arg in mp_dict['exception'].args:
        msg += str(e_arg)
    print(str(msg), flush=True)
    pytest.fail(str(msg), pytrace=False)


def run_workers(func, world_size):
    """
    Starts `world_size` number of processes, where each of them
    behaves as a separate worker and invokes function specified 
    by the parameter.
    
    :param func: The function that will be invoked by each process
    :type func: function
    :returns: mp_dict which can be used by workers to return
    results from `func`
    """
    manager = mp.Manager()
    mp_dict = manager.dict()
    processes = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for rank in range(1, world_size):
            p = mp.Process(target=func, args=(mp_dict, rank, world_size, tmp_dir))
            p.daemon = True
            p.start()
            processes.append(p)
        func(mp_dict, 0, world_size, tmp_dir)
            
        for p in processes:
            p.join()
        if 'exception' in mp_dict:
            handle_mp_exception(mp_dict)
    return mp_dict

    
def sar_test(func):
    """
    A decorator function that wraps all SAR tests with the primary objective
    of facilitating module imports in tests without affecting other tests.

    :param func: The function that serves as the entry point to the test.
    :type func: function
    :returns: A function that encapsulates the pytest function.
    """
    @functools.wraps(func)
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
        p = mp.Process(target=process_wrapper, args=mp_args, kwargs=kwargs)
        p.start()
        p.join()

        if 'exception' in mp_dict:
            handle_mp_exception(mp_dict)

        return mp_dict["result"]
    return test_wrapper
