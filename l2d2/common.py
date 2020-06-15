import contextlib
import datetime
from typing import Dict, Tuple

import termcolor


@contextlib.contextmanager
def scoped_time(annotation: str):
    print('...', annotation)
    start = datetime.datetime.now()
    yield
    end = datetime.datetime.now()
    print('...', annotation, 'done in', end-start)


def print_confusion(confusion: Dict[Tuple[int, int], int], classes: int) -> float:
    # Print out the confusion matrix.
    for want in range(classes):
        print(f'want {want:2d}: ', end='')
        for got in range(classes):
            value = confusion[(want, got)]
            color = 'green' if want == got else ('red' if value != 0 else None)
            print(termcolor.colored(f'{value:5d}', color=color), end=' ')
        print()

    # Print out summary accuracy statistic(s).
    correct = sum(
            confusion[(i, i)]
            for i in range(classes))
    total = sum(confusion.values())
    accuracy = correct / float(total) * 100.0
    print(f'accuracy: {accuracy:.2f}% ({correct} / {total})')
    return accuracy
