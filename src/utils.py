import time


class Timer:

    def start(self):
        self.start = time.time()

    def end(self):
        self.end = time.time()
        print('Done in {:.2f}s'.format(self.end - self.start))


def progress_bar(progress, total, message='', size=20):
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
    complete = str(''.join('-' * cur))
    incomplete = str(''.join(' ' * (size - cur)))

    # Print the progress bar
    print("\r" + '[' + complete + incomplete + '] {:.2f}% complete'.format(perc * 100), message, end="")