import sys
import re


# https://stackoverflow.com/questions/24204898/python-output-on-both-console-and-file
# https://stackoverflow.com/questions/46971566/python-to-handle-color-to-terminal-and-also-in-file
class Logger:
    def __init__(self, filename):
        self.out_file = open(filename, "w")
        self.old_stdout = sys.stdout
        #this object will take over `stdout`'s job
        sys.stdout = self
        self.start_logging = False

    #executed when the user does a `print`
    def write(self, text):
        self.old_stdout.write(text)
        plain_message = re.sub('\033\\[\d+m', '', text)
        if re.match('\[.+\]', plain_message):
            self.start_logging = True
        if self.start_logging :
            self.out_file.write(re.sub('\033\\[\d+m', '', text))
        if "\n" in plain_message:
            self.start_logging = False

    #executed when `with` block begins
    def __enter__(self):
        return self

    #executed when `with` block ends
    def __exit__(self, type, value, traceback):
        #we don't want to log anymore. Restore the original stdout object.
        sys.stdout = self.old_stdout
