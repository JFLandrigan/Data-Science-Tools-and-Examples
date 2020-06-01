import os
import sys
import pandas as pd

if "win" in sys.platform:
    ext_char = "\\"
else:
    ext_char = "/"

file_path = os.path.abspath(os.path.dirname(__file__)) + ext_char


def load_iris():
    return pd.read_csv(file_path + "iris.csv")


def load_stack_overflow_dat():
    return pd.read_csv(file_path + "stack-overflow-data.csv")
