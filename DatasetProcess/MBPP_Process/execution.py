import json
import sys
import subprocess
from subprocess import TimeoutExpired, CalledProcessError
import os
import shutil
from tempfile import mkdtemp
from contextlib import contextmanager
import multiprocessing
import platform
import resource

@contextmanager
def create_tempdir():
    dirname = mkdtemp()
    try:
        yield dirname
    finally:
        shutil.rmtree(dirname)


def reliability_guard():
    # 禁用一些可能危险的函数
    dangerous_functions = [
        'os.kill', 'os.system', 'os.putenv', 'os.remove', 'os.removedirs',
        'shutil.rmtree', 'shutil.move', 'subprocess.Popen', 'sys.exit'
    ]
    for func in dangerous_functions:
        globals()[func] = None


def run_script(script, timeout):
    with create_tempdir() as dirname:
        script_path = os.path.join(dirname, 'temp_script.py')
        with open(script_path, 'w') as f:
            f.write(script)

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                timeout=timeout,
                check=True
            )
            return "All test cases passed."
        except TimeoutExpired:
            return "Test timeout after {} seconds.".format(timeout)
        except CalledProcessError as e:
            return f"Test failed with error:\n{e.stderr.decode()}"


def worker(script, timeout):
    reliability_guard()
    return run_script(script, timeout)


def check_correctness(problem, code, timeout=5, completion_id=None):
    script = code + "\n" + "\n".join(
        problem['test_list'])

    print(script)

    pool = multiprocessing.Pool(processes=1)
    result = pool.apply_async(worker, (script, timeout))
    try:
        status = result.get(timeout=timeout * 1.1)
        passed = status == "All test cases passed."
    except multiprocessing.TimeoutError:
        status = "Test did not complete in the given time."
        passed = False

    # 定义测试结果字典
    test_result = dict(
        task_id=problem["task_id"],
        passed=passed,
        result=status,
        completion_id=completion_id,
    )

    return test_result