import os
import sys
import numpy as np
import datetime
import time
import psutil


def monitor_usage(log_file: str, interval: float = 10):
    """Continously monitor CPU usage."""
    ppid = os.getppid()
    if not os.path.isfile(log_file):
        with open(log_file, 'w') as f:
            f.write(f'Log file for {ppid}\n')
    while True:
        cpu_usage = psutil.cpu_percent(percpu=True)
        mean_cpu_usage = np.mean(cpu_usage)
        n_cores = len(cpu_usage)
        time_now = datetime.datetime.now()
        message = (
            f'{time_now} CPU usage {mean_cpu_usage:.2f}% '
            f'({n_cores} cores)'
        )
        with open(log_file, 'a') as f:
            f.write(message + '\n')
        time.sleep(interval)


if __name__ == '__main__':
    monitor_usage(sys.argv[1])
