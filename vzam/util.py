from sklearn.preprocessing import normalize
import subprocess
import os


def normalize_l1(arr):
    return normalize(arr.reshape(1, -1), 'l1').flatten()


def run_command(cmd):
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True,
    )
    stderr_text = p.stderr.read()
    if stderr_text:
        raise Exception(stderr_text)

    stdout_text = p.stdout.read()
    return stdout_text


def ffmpeg(options):
    command = f'ffmpeg {options}'
    return run_command(command)


def mkdir_safe(path):
    if not os.path.exists(path):
        os.mkdir(path)