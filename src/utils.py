import time


class Timer:

    def start(self):
        self.start = time.time()

    def end(self):
        self.end = time.time()
        print('Done in {:.2f}s'.format(self.end - self.start))


class Color:

    template = '\033[38;2;{};{};{}m'
    clear = '\033[0m'

    green =(0, 175, 95)
    red = (175, 0, 0)
    white = (185,185,185)
    black = (8,8,8)

    def __init__(self, rgb):
        self.prefix = Color.template.format(rgb[0], rgb[1], rgb[2])

    def paint(self, string):
        return self.prefix + string + self.clear


def progress_bar(progress, total, message='', size=20, color=True):
    """
    Prints a progress bar. Overwrites the last progress bar written using the "\r" (carriage return) character.
    :param progress: The current progress, i.e. the number of things
    :param total: The total number of things you have to do to reach 100.00%
    :param message: An optional argument: the extra message to display inline.
    :param size: Optionally, the size of the progress bar to display, in characters.
    """
    # Check some error conditions
    if total <= 0 or size < 0:
        return
    if progress < 0:
        progress = 0

    # Do some calculations
    perc = float(progress) / float(total)
    cur = int(perc * size) if progress < total else int(size)

    # Make the progress strings
    complete = str(''.join(chr(9608) * cur))
    incomplete = str(''.join(' ' * (size - cur)))

    # Get the linear interpolation between two colors at the current percentage
    if color:
        t = tuple(map(lambda c: int(c[0] * (1 - perc) + c[1] * perc), zip(Color.black, Color.white)))
        c = Color(t)
        complete = c.paint(complete)

    # Print the progress bar
    print("\r" + '|' + complete + incomplete + '| '    # [██████████████████████████████████████████████████]
          '{:.2f}% complete'.format(perc * 100),                # 100.00% complete
          message, end="")


if __name__ == '__main__':

    for i in range(101):
        progress_bar(i, 100, size=50)
        time.sleep(.03)