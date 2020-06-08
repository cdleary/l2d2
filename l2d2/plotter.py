import datetime
import fileinput
import re
from dataclasses import dataclass
from typing import List

from matplotlib import pyplot as plt


LINE_RE = re.compile(r't: (?P<t>.*?) epochno: (?P<epochno>\d+) stepno: (?P<stepno>\d+) accuracy: (?P<accuracy>[\d.]+) loss: (?P<loss>[\d.]+)')


@dataclass
class Data:
    #t: List[datetime.datetime]
    #epochno: List[int]
    stepno: List[int]
    accuracy: List[float]
    loss: List[float]


def get_data() -> Data:
    d = Data([], [], [])
    for line in fileinput.input():
        m = LINE_RE.match(line)
        stepno = int(m.group('stepno'))
        accuracy = float(m.group('accuracy'))
        loss = float(m.group('loss'))
        d.stepno.append(stepno)
        d.accuracy.append(accuracy)
        d.loss.append(loss)
    return d


def _get_data_and_plot():
    d = get_data()
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(d.stepno, d.accuracy, '.', label='accuracy', color=color)
    ax1.set_ylabel('accuracy', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('loss', color=color)
    ax2.plot(d.stepno, d.loss, '.', label='loss', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.legend()
    fig.tight_layout()
    plt.show()


def main():
    _get_data_and_plot()


if __name__ == '__main__':
    main()
