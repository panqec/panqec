"""
Routines for checking usage and progress.

Works by reading usage txt files produced by the `monitor` command.
"""

from typing import List, Dict, Tuple, Any, Optional, Union
import os
import re
from glob import glob
import shutil
import numpy as np
import pandas as pd


def summarize_usage(logs_dirs: List[str]):
    print(f'Checking usage in {len(logs_dirs)} directories')
    usage_df_list = []
    for logs_dir in logs_dirs:
        jobs = get_jobs(logs_dir)
        usage = get_usage(logs_dir, jobs)
        usage_df_list.append(usage)
    if usage_df_list:
        usage_df = pd.concat(usage_df_list, axis=0)
        time_df = get_time_df(usage_df)
        plot_usage(time_df)
    else:
        print('No usage logs found')


def plot_usage(time_df):
    plotter = TextPlotter(height=20)
    plotter.fill(time_df['hour'], time_df['cores_avail'], '.', label='Avail')
    plotter.fill(time_df['hour'], time_df['cores_used'], '#', label='Used')
    plotter.ylabel('CPU cores')
    plotter.title('Resource usage over time')
    plotter.legend()
    plotter.show()

    plotter = TextPlotter(height=20)
    plotter.fill(time_df['hour'], time_df['ram_avail'], '.', label='Avail')
    plotter.fill(time_df['hour'], time_df['ram_used'], '#', label='Used')
    plotter.xlabel('Time (h)')
    plotter.ylabel('RAM (GiB)')
    plotter.legend()
    plotter.show()


def get_jobs(logs_dir):
    jobs_list = []
    usage_files = glob(os.path.join(logs_dir, 'usage_*.txt'))

    for usage_file in usage_files:
        file_name = os.path.split(usage_file)[-1]
        match = re.search(r'usage_(\d+)_(\d+).txt', file_name)
        node_id = match.group(1)
        array_index = match.group(2)
        filter_file = os.path.join(
            logs_dir, f'filter_{node_id}_{array_index}.txt'
        )
        if not os.path.isfile(filter_file):
            filter_file = None
        jobs_list.append({
            'node_id': node_id,
            'array_index': array_index,
            'usage_file': usage_file,
            'filter_file': filter_file,
        })
    jobs = pd.DataFrame(jobs_list)

    jobs['array_index'] = jobs['array_index'].astype('int')
    jobs = jobs.sort_values(by='array_index').reset_index(drop=True)
    return jobs


def get_usage(logs_dir, jobs):
    usage_pattern = re.compile(
        r'^(?P<date>[\d\-\:\.\s]+) '
        r'CPU usage (?P<cpu_pc>[\d\.]+)% \((?P<n_cores>\d+) cores\) '
        r'RAM (?P<ram_pc>[\d.]+)% \((?P<tot_mem>[\d\.]+) GiB tot\)'
    )
    usage_list = []
    for _, job in jobs.iterrows():
        with open(job['usage_file']) as f:
            lines = f.readlines()
        for line in lines:
            usage_match = re.search(usage_pattern, line)
            if usage_match:
                datapoint = {
                    k: usage_match.group(k)
                    for k in usage_pattern.groupindex
                }
                datapoint['node_id'] = job['node_id']
                usage_list.append(datapoint)
    usage = pd.DataFrame(usage_list)
    usage['date'] = pd.to_datetime(usage['date'])
    for column, dtype in [
        ('cpu_pc', float), ('n_cores', int),
        ('ram_pc', float), ('tot_mem', float)
    ]:
        usage[column] = usage[column].astype(dtype)
    usage['cores'] = usage['cpu_pc']*usage['n_cores']/100
    usage['ram'] = usage['ram_pc']*usage['tot_mem']/100
    return usage


def get_time_df(usage):
    date_ranges = pd.date_range(
        usage.date.min(), usage.date.max(), periods=100
    )
    cores_used = [0]
    cores_avail = [0]
    ram_used = [0]
    ram_avail = [0]
    for i in range(len(date_ranges) - 1):
        usage_filt = usage[
            (date_ranges[i] <= usage.date)
            & (usage.date <= date_ranges[i + 1])
        ].groupby('node_id')
        cores_used.append(usage_filt['cores'].max().sum())
        cores_avail.append(usage_filt['n_cores'].max().sum())
        ram_used.append(usage_filt['ram'].max().sum())
        ram_avail.append(usage_filt['tot_mem'].max().sum())

    hours = (date_ranges - date_ranges[0]).total_seconds()/60/60

    time_df = pd.DataFrame({
        'time': date_ranges,
        'hour': hours,
        'cores_used': cores_used,
        'cores_avail': cores_avail,
        'ram_used': ram_used,
        'ram_avail': ram_avail,
    })
    return time_df


class TextPlotter:
    """
    ASCII plotter like matplotlib but in command line.
    """

    canvas_width: int
    canvas_height: int

    def __init__(
        self, width: Optional[int] = None, height: Optional[int] = None
    ):
        term = shutil.get_terminal_size()
        if height is None:
            self.canvas_height = term.lines - 1
        else:
            self.canvas_height = height
        if width is None:
            self.canvas_width = term.columns - 1
        else:
            self.canvas_width = width
        self._xlim: List[Optional[float]] = [None, None]
        self._ylim: List[Optional[float]] = [None, None]
        self._xlabel: Optional[str] = None
        self._ylabel: Optional[str] = None
        self._data: List[Dict[str, Any]] = []
        self._title: Optional[str] = None
        self._legend: bool = False

    def title(self, label) -> None:
        self._title = label

    def _blank_canvas(self) -> None:
        self.lines = [
            [' ' for j in range(self.canvas_width)]
            for i in range(self.canvas_height)
        ]

    def _draw_axes(self) -> None:
        for j in range(self.plot_height):
            self.lines[self.top_margin + j][self.left_margin] = '|'
            self.lines[self.top_margin + j][
                self.canvas_width - self.right_margin - 1
            ] = '|'
        for j in range(self.plot_width):
            self.lines[
                self.canvas_height - self.bottom_margin
            ][j + self.left_margin] = '-'
            self.lines[
                self.top_margin
            ][j + self.left_margin] = '-'
        self.lines[
            self.canvas_height - self.bottom_margin
        ][self.left_margin] = '+'
        self.lines[
            self.canvas_height - self.bottom_margin
        ][self.canvas_width - self.right_margin - 1] = '+'
        self.lines[self.top_margin][self.left_margin] = '+'
        self.lines[self.top_margin][
            self.canvas_width - self.right_margin - 1
        ] = '+'

    def _init_sizes(self) -> None:
        self.left_margin = 5
        if self._ylabel is not None:
            self.left_margin = min(len(self._ylabel) + 2, 20)

        self.right_margin = 2
        if self._legend:
            legend_rows = self._split_text(self._get_legend_str(), wrap=20)
            legend_width = max(len(row) for row in legend_rows)
            self.right_margin = max(5, legend_width + 2)

        self.bottom_margin = 2
        if self._xlabel:
            self.bottom_margin = 4

        self.plot_width = (
            self.canvas_width - self.left_margin - self.right_margin
        )

        self.top_margin = 1
        if self._title:
            n_title_rows = len(
                self._split_text(self._title, wrap=self.plot_width)
            )
            self.top_margin = max(1, n_title_rows + 1)

        self.plot_height = (
            self.canvas_height - self.bottom_margin - self.top_margin
        )

    def _split_text(self, text: str, wrap: int) -> List[str]:
        split_text = []
        for line in text.split('\n'):
            if len(line) <= wrap:
                split_text.append(line)
            else:
                words = line.split(' ')
                subline = ''
                while len(words) > 0:
                    next_word = words.pop(0)
                    if len(next_word) >= wrap:
                        split_text.append(subline)
                        split_text.append(next_word)
                        subline = ''
                    elif len(subline) + len(next_word) >= wrap:
                        split_text.append(subline)
                        subline = next_word
                    else:
                        if subline:
                            subline += ' ' + next_word
                        else:
                            subline = next_word
                if subline:
                    split_text.append(subline)
        return split_text

    def _label_text(
        self, label: str, x: int, y: int, vertical: bool = False,
        halign: str = 'left', valign: str = 'top',
        wrap: Optional[int] = None
    ) -> None:

        if wrap is None:
            if vertical:
                wrap = self.canvas_height
            else:
                wrap = self.canvas_width

        # Split into rows.
        label_rows = self._split_text(label, wrap=wrap)
        n_rows = len(label_rows)

        voffset = 0
        if n_rows > 1:
            if valign == 'center':
                voffset = int(-len(label_rows)/2)
            elif valign == 'bottom':
                voffset = int(-len(label_rows))

        if vertical:
            if x + voffset <= 0:
                voffset = -x
            elif x + n_rows + voffset >= self.canvas_width:
                voffset = self.canvas_width - x - n_rows
        else:
            if y + voffset <= 0:
                voffset = -y
            elif y + n_rows + voffset >= self.canvas_height:
                voffset = self.canvas_height - y - n_rows

        for i, row in enumerate(label_rows):
            row_width = len(row)

            # Offset depending on alignment.
            hoffset = 0
            if halign == 'right':
                hoffset = -row_width
            elif halign == 'center':
                hoffset = int(-row_width/2)

            if vertical:

                # Ensure text within boundaries.
                if y + hoffset <= 0:
                    hoffset = -y
                elif y + hoffset + row_width >= self.canvas_height:
                    hoffset = self.canvas_height - y - hoffset - row_width

                for j, char in enumerate(row):
                    self.lines[y + j + hoffset][x + i + voffset] = char
            else:

                # Ensure text within boundaries.
                if x + hoffset <= 0:
                    hoffset = -x
                elif x + hoffset + row_width > self.canvas_width:
                    hoffset = self.canvas_width - x - hoffset - row_width - 2

                for j, char in enumerate(row):
                    self.lines[y + i + voffset][j + x + hoffset] = char

    def xlabel(self, label: str) -> None:
        self._xlabel = label

    def ylabel(self, label: str) -> None:
        self._ylabel = label

    def _draw_xlabel(self) -> None:
        if self._xlabel is not None:
            x = int(self.left_margin + self.plot_width/2)
            y = int(self.canvas_height - self.bottom_margin/2)
            self._label_text(self._xlabel, x, y, halign='center')

    def _draw_ylabel(self) -> None:
        if self._ylabel is not None:
            max_width = max(
                len(line)
                for line in self._split_text(
                    self._ylabel, wrap=self.left_margin - 1
                )
            )
            if max_width < self.left_margin:
                x = int(self.left_margin/2)
                y = int(self.top_margin + self.plot_height/2)
                self._label_text(
                    self._ylabel, x, y, vertical=False,
                    halign='center', valign='center',
                    wrap=self.left_margin - 1
                )
            else:
                x = int(self.left_margin/2)
                y = int(self.top_margin + self.plot_height/2)
                self._label_text(
                    self._ylabel, x, y, vertical=True,
                    halign='center', valign='center'
                )

    def fill(
        self, x: np.ndarray, y: np.ndarray, marker: str = '.',
        label: Optional[str] = None
    ) -> None:
        self._data.append({
            'linetype': 'fill',
            'x': x,
            'y': y,
            'marker': marker,
            'label': label,
        })

    def plot(
        self, x: np.ndarray, y: np.ndarray, marker: str = '.',
        label: Optional[str] = None
    ) -> None:
        self._data.append({
            'linetype': 'plot',
            'x': x,
            'y': y,
            'marker': marker,
            'label': label,
        })

    def _get_limits(self) -> Tuple[float, float, float, float]:
        x_lim_min: float = self._xlim[0] if self._xlim[0] is not None else 0
        x_lim_max: float = self._xlim[1] if self._xlim[1] is not None else 1
        y_lim_min: float = self._ylim[0] if self._ylim[0] is not None else 0
        y_lim_max: float = self._ylim[1] if self._ylim[1] is not None else 1
        return x_lim_min, x_lim_max, y_lim_min, y_lim_max

    def _draw_fill(self) -> None:
        x_lim_min, x_lim_max, y_lim_min, y_lim_max = self._get_limits()
        for series in self._data:
            if series['linetype'] == 'fill':
                x = np.array(series['x'])
                y = np.array(series['y'])
                marker = series['marker']
                x_spacing = (x_lim_max - x_lim_min)/self.plot_width
                y_spacing = (y_lim_max - y_lim_min)/self.plot_height
                for i in range(self.plot_width):
                    x_min = x_lim_min + i*x_spacing
                    x_max = x_lim_min + (i + 1)*x_spacing
                    index = np.where((x_min <= x) & (x <= x_max))[0]
                    if len(index) > 0:
                        for j in range(self.plot_height):
                            y_row = y_lim_min + j*y_spacing
                            y_max = np.max(y[index])
                            if y_row <= y_max:
                                self.lines[
                                    self.canvas_height - self.bottom_margin - j
                                ][i + self.left_margin] = marker

    def _draw_plot(self) -> None:
        x_lim_min, x_lim_max, y_lim_min, y_lim_max = self._get_limits()
        for series in self._data:
            if series['linetype'] == 'plot':
                x = np.array(series['x'])
                y = np.array(series['y'])
                marker = series['marker']
                x_spacing = (x_lim_max - x_lim_min)/self.plot_width
                y_spacing = (y_lim_max - y_lim_min)/self.plot_height
                for i in range(self.plot_width):
                    for j in range(self.plot_height):
                        x_min = x_lim_min + i*x_spacing
                        x_max = x_lim_min + (i + 1)*x_spacing
                        y_min = y_lim_min + j*y_spacing
                        y_max = y_lim_min + (j + 1)*y_spacing
                        if np.any(
                            (x_min <= x) & (x <= x_max)
                            & (y_min <= y) & (y <= y_max)
                        ):
                            self.lines[
                                self.canvas_height - self.bottom_margin - j
                            ][i + self.left_margin] = marker

    def legend(self) -> None:
        self._legend = True

    def _format_number(self, value: Optional[Union[int, float]]) -> str:
        if value is None:
            return ''
        elif value % 1 == 0:
            return str(int(value))
        else:
            return '{:.4g}'.format(value)

    def _label_limits(self) -> None:
        self._label_text(
            self._format_number(self._xlim[0]),
            self.left_margin, self.top_margin + self.plot_height + 1,
            halign='left', valign='top'
        )
        self._label_text(
            self._format_number(self._xlim[1]),
            self.canvas_width - self.right_margin,
            self.top_margin + self.plot_height + 1,
            halign='center', valign='top'
        )
        self._label_text(
            self._format_number(self._ylim[0]),
            self.left_margin - 1, self.top_margin + self.plot_height,
            halign='right', valign='bottom'
        )
        self._label_text(
            self._format_number(self._ylim[1]),
            self.left_margin - 1, self.top_margin,
            halign='right', valign='top'
        )

    def _draw_title(self) -> None:
        if self._title is not None:
            self._label_text(
                self._title,
                int(self.left_margin + self.plot_width/2),
                int(self.top_margin/2),
                halign='center', valign='center'
            )

    def _set_limits(self) -> None:
        if self._xlim[0] is None:
            self._xlim[0] = min([np.min(series['x']) for series in self._data])
        if self._xlim[1] is None:
            self._xlim[1] = max([np.max(series['x']) for series in self._data])
        if self._ylim[0] is None:
            self._ylim[0] = min([np.min(series['y']) for series in self._data])
        if self._ylim[1] is None:
            self._ylim[1] = max([np.max(series['y']) for series in self._data])

    def ylim(self, y_min: float, y_max: float) -> None:
        self._ylim = [y_min, y_max]

    def xlim(self, x_min: float, x_max: float) -> None:
        self._xlim = [x_min, x_max]

    def _get_legend_str(self) -> str:
        legend_strings = []
        for series in self._data:
            if 'label' in series and series['label'] is not None:
                label = series['label']
                marker = series['marker']
                legend_strings.append(marker*3 + ' ' + label)
        return '\n'.join(legend_strings)

    def _draw_legend(self) -> None:
        if self._legend:
            self._label_text(
                self._get_legend_str(),
                self.canvas_width - self.right_margin + 2,
                int(self.canvas_height/2),
                halign='left', valign='center',
                wrap=self.right_margin
            )

    def render(self) -> str:
        self._init_sizes()
        self._blank_canvas()
        self._set_limits()
        self._draw_fill()
        self._draw_plot()
        self._draw_axes()
        self._draw_xlabel()
        self._draw_ylabel()
        self._draw_title()
        self._draw_legend()
        self._label_limits()
        return '\n'.join([''.join(line) for line in self.lines])

    def show(self) -> None:
        render = self.render()
        print(render)
