import contextlib
import datetime


@contextlib.contextmanager
def scoped_time(annotation):
    print('...', annotation)
    start = datetime.datetime.now()
    yield
    end = datetime.datetime.now()
    print('...', annotation, 'done in', end-start)


