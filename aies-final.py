import random

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib

import seaborn as sns
from scipy.stats import pearsonr
from itertools import cycle


def STEP1():
    df_por = pd.read_csv('dataset/student-por.csv', delimiter=';')
    print(df_por.head(5))
    print("Number of records port: ", len(df_por))
    r, c = df_por.shape
    print("rows port: ", r)
    print("columns port: ", c)
    print()
    for col in df_por.columns:
        print(col)

    # df_mat = pd.read_csv('dataset/student-mat.csv', delimiter=';')
    # print(df_mat.head(5))
    # print("Number of records math: ", len(df_mat))
    # r, c = df_mat.shape
    # print("rows math: ", r)
    # print("columns math: ", c)
    # print()
    # for col in df_mat.columns:
    #     print(col)
    #
    # merged = [df_por, df_mat]
    # merged = pd.concat(merged)
    # print(merged.head(5))
    # print("Number of records merged: ", len(merged))
    # r, c = merged.shape
    # print("rows merged: ", r)
    # print("columns merged: ", c)
    # print()
    # for col in merged.columns:
    #     print(col)
    # merged.to_csv(r'out/merged.csv')


def STEP2():
    df = pd.read_csv(r"dataset/", delimiter=',')
    print("Number of records: ")
    r, c = df.shape
    print("rows: ", r)
    print("columns: ", c)
    print()


def STEP3():
    df = pd.read_csv(r"dataset/", delimiter=',')

    print("Number of records: ")
    r, c = df.shape
    print("rows: ", r)
    print("columns: ", c)
    print()


def STEP4():
    df = pd.read_csv(r"dataset/", delimiter=',')

    print("Number of records: ")
    r, c = df.shape
    print("rows: ", r)
    print("columns: ", c)
    print()


if __name__ == '__main__':
    STEP1()
    # STEP2()
    # STEP3()
    # STEP4()
