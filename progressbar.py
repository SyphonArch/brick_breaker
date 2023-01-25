class ProgressBar:
    def __init__(self, n: int, length: int = 32, char: str = '|'):
        """Create a ProgressBar.

        n is the number of calls required to reach 100%
        length is the visual length of the bar, in characters.
        char is the character used to display the progress bar.
        """
        assert len(char) == 1
        self.n = n
        self.length = length
        self.char = char
        self.i = 0

    def __repr__(self):
        progress = self.i / self.n
        percentage = int(progress * 100)
        bar_length = int(progress * self.length)
        return f"[{self.char * bar_length}{' ' * (self.length - bar_length)}] " \
               f"[{percentage:3}%] ({self.i} /{self.n})"

    def start(self):
        print(self.__repr__(), end='')

    def update(self, *_):
        self.i += 1
        print('\r' + self.__repr__(), end='')
        if self.i == self.n:
            print()
