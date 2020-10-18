from time import time

stack = []
index = dict()
flags = []
levels = []
timings = []


class Timer(object):
    def __init__(self, description):
        self.description = description

    def __enter__(self):
        stack.append(self.description)
        if self.description not in index:
            index[self.description] = len(flags)
            flags.append(self.description)
            levels.append(len(stack))
            timings.append(0.0)
        id = index[self.description]
        timings[id] -= time()

    def __exit__(self, type, value, traceback):
        stack.pop()
        id = index[self.description]
        timings[id] += time()


def Timer_Print():
    total = 0.0
    for l, t in zip(levels, timings):
        if l == 1:
            total += t
    print('')
    for f, l, t in zip(flags, levels, timings):
        print('  ' * l, end='')
        print('{0:s} : {1:4f} ({2:.0%})'.format(f, t, t / total))
    print('')
