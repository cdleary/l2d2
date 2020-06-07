import datetime
import fileinput
import re
from dataclasses import dataclass
from typing import List

from matplotlib import pyplot as plt


LINE_RE = re.compile(r't: (?P<t>.*?) epochno: (?P<epochno>\d+) stepno: (?P<stepno>\d+) accuracy: (?P<accuracy>[\d.]+)')


@dataclass
class Data:
    #t: List[datetime.datetime]
    #epochno: List[int]
    stepno: List[int]
    accuracy: List[float]


def get_data() -> Data:
    d = Data([], [])
    for line in fileinput.input():
        m = LINE_RE.match(line)
        stepno = int(m.group('stepno'))
        accuracy = float(m.group('accuracy'))
        d.stepno.append(stepno)
        d.accuracy.append(accuracy)
    return d


def main():
    d = get_data()
    plt.plot(d.stepno, d.accuracy, '.')
    plt.show()


if __name__ == '__main__':
    main()
