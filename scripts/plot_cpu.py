import sys
import re
import subprocess
import numpy as np


def main():
    usage_txt_list = sys.argv[1:]
    for usage_txt in usage_txt_list:
        y = np.array(get_cpu_usage(usage_txt))
        x = np.arange(len(y))
        plot_points(x, y)


def plot_points(x, y):
    gnuplot = subprocess.Popen(
        ["/usr/bin/gnuplot"],
        stdin=subprocess.PIPE
    )
    gnuplot.stdin.write("set term dumb 79 25\n".encode())
    gnuplot.stdin.write(
        "plot '-' using 1:2 title 'CPU Usage (%)' with linespoints \n".encode()
    )
    for i, j in zip(x, y):
        gnuplot.stdin.write('{} {}\n'.format(i, j).encode())
    gnuplot.stdin.write("e\n".encode())
    gnuplot.stdin.flush()
    gnuplot.stdin.write("\n".encode())


def get_cpu_usage(usage_txt):
    with open(usage_txt) as f:
        lines = f.readlines()
    points = []
    for line in lines:
        match = re.search(r'CPU usage ([\d\.]+)%', line)
        if match:
            points.append(match.group(1))
    return points


if __name__ == '__main__':
    main()
