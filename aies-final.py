import random

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy.stats import pearsonr
from itertools import cycle


def STEP1():
    df = pd.read_csv(r"/Users/Yaima/Documents/Python/AIES/toxity_per_attribute.csv", delimiter=',')

    print("Number of records: ")
    r, c = df.shape
    print("rows: ", r)
    print("columns: ", c)
    print()

def STEP2():
    df = pd.read_csv(r"/Users/Yaima/Documents/Python/AIES/toxity_per_attribute.csv", delimiter=',')

    print("Number of records: ")
    r, c = df.shape
    print("rows: ", r)
    print("columns: ", c)
    print()


def STEP3():
    df = pd.read_csv(r"/Users/Yaima/Documents/Python/AIES/toxity_per_attribute.csv", delimiter=',')

    print("Number of records: ")
    r, c = df.shape
    print("rows: ", r)
    print("columns: ", c)
    print()

def STEP4():
    df = pd.read_csv(r"/Users/Yaima/Documents/Python/AIES/toxity_per_attribute.csv", delimiter=',')

    print("Number of records: ")
    r, c = df.shape
    print("rows: ", r)
    print("columns: ", c)
    print()

if __name__ == '__main__':
    STEP1()
    STEP2()
    STEP3()
    STEP4()
    STEP5()
