import ctypes
import os

so = ctypes.CDLL(os.path.dirname(os.path.realpath(__file__)) + "/wrapper/a.so")
