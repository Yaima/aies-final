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
    df = pd.read_csv('dataset/student-por.csv', delimiter=',')
    print(df.head(5))
    print("Number of records port: ", len(df))
    r, c = df.shape
    print("rows port: ", r)
    print("columns port: ", c)
    print()
    for col in df.columns:
        print(col)


def STEP2():
    df = pd.read_csv('dataset/student-por.csv', delimiter=',')

    df.loc[df['sex'] == "M", ['sex']] = 'Male'
    df.loc[df['sex'] == "F", ['sex']] = 'Female'

    df.loc[df['age'].between(15, 16, inclusive=True), ['age_group']] = '15-16'
    df.loc[df['age'].between(17, 18, inclusive=True), ['age_group']] = '17-18'
    df.loc[df['age'].between(19, 22, inclusive=True), ['age_group']] = '19-22'

    age_group_sex_freq = df.groupby(['age_group', 'sex']).size()
    age_group_sex_freq.to_csv('out/age_group_sex_freq.csv')

    age_unique = []
    g1_unique = []
    g2_unique = []
    g3_unique = []

    for x in df['age']:
        if x not in age_unique:
            age_unique.append(x)

    for x in df['G1']:
        if x not in g1_unique:
            g1_unique.append(x)

    for x in df['G2']:
        if x not in g2_unique:
            g2_unique.append(x)

    for x in df['G3']:
        if x not in g3_unique:
            g3_unique.append(x)

    print("Age: ", age_unique)
    print("G1: ", g1_unique)
    print("G2: ", g2_unique)
    print("G3: ", g3_unique)

    print("G1 Max: ", df['G1'].max())
    print("G1 Min: ", df['G1'].min())
    print("G1 Avg: ", df['G1'].mean())

    print("G2 Max: ", df['G2'].max())
    print("G2 Min: ", df['G2'].min())
    print("G2 Avg: ", df['G2'].mean())

    print("G3 Max: ", df['G3'].max())
    print("G3 Min: ", df['G3'].min())
    print("G3 Avg: ", df['G3'].mean())

    df.loc[df['G1'].between(0, 5, inclusive=True), ['g1_group']] = '0-5'
    df.loc[df['G1'].between(6, 10, inclusive=True), ['g1_group']] = '6-10'
    df.loc[df['G1'].between(11, 15, inclusive=True), ['g1_group']] = '11-15'
    df.loc[df['G1'].between(16, 19, inclusive=True), ['g1_group']] = '16-19'

    df.loc[df['G2'].between(0, 5, inclusive=True), ['g2_group']] = '0-5'
    df.loc[df['G2'].between(6, 10, inclusive=True), ['g2_group']] = '6-10'
    df.loc[df['G2'].between(11, 15, inclusive=True), ['g2_group']] = '11-15'
    df.loc[df['G2'].between(16, 19, inclusive=True), ['g2_group']] = '16-19'

    df.loc[df['G3'].between(0, 5, inclusive=True), ['g3_group']] = '0-5'
    df.loc[df['G3'].between(6, 10, inclusive=True), ['g3_group']] = '6-10'
    df.loc[df['G3'].between(11, 15, inclusive=True), ['g3_group']] = '11-15'
    df.loc[df['G3'].between(16, 19, inclusive=True), ['g3_group']] = '16-19'

    age_g1_freq = df.groupby(['age_group', 'g1_group']).size()
    age_g1_freq.to_csv('out/age_g1_freq.csv')

    age_g2_freq = df.groupby(['age_group', 'g2_group']).size()
    age_g2_freq.to_csv('out/age_g2_freq.csv')

    age_g3_freq = df.groupby(['age_group', 'g3_group']).size()
    age_g3_freq.to_csv('out/age_g3_freq.csv')

    sex_g1_freq = df.groupby(['sex', 'g1_group']).size()
    sex_g1_freq.to_csv('out/sex_g1_freq.csv')

    sex_g2_freq = df.groupby(['sex', 'g1_group']).size()
    sex_g2_freq.to_csv('out/sex_g2_freq.csv')

    sex_g3_freq = df.groupby(['sex', 'g2_group']).size()
    sex_g3_freq.to_csv('out/sex_g3_freq.csv')

    age_g1_freq.plot.bar(stacked=False, color="royalblue")
    plt.title('Age - Grade 1 Frequencies')
    plt.grid(True, axis='y', alpha=0.2, color='#999999')
    plt.xlabel('Frequency Groups')
    plt.savefig('out/age_g1_freq.png', bbox_inches='tight')
    plt.show()

    age_g2_freq.plot.bar(stacked=False, color="royalblue")
    plt.title('Age - Grade 2 Frequencies')
    plt.grid(True, axis='y', alpha=0.2, color='#999999')
    plt.xlabel('Frequency Groups')
    plt.savefig('out/age_g2_freq.png', bbox_inches='tight')
    plt.show()

    age_g3_freq.plot.bar(stacked=False, color="royalblue")
    plt.title('Age - Grade 3 Frequencies')
    plt.grid(True, axis='y', alpha=0.2, color='#999999')
    plt.xlabel('Frequency Groups')
    plt.savefig('out/age_g3_freq.png', bbox_inches='tight')
    plt.show()

    sex_g1_freq.plot.bar(stacked=False, color="lightblue")
    plt.title('Sex - Grade 1 Frequencies')
    plt.grid(True, axis='y', alpha=0.2, color='#999999')
    plt.xlabel('Frequency Groups')
    plt.savefig('out/sex_g1_freq.png', bbox_inches='tight')
    plt.show()

    sex_g2_freq.plot.bar(stacked=False, color="lightblue")
    plt.title('Sex - Grade 2 Frequencies')
    plt.grid(True, axis='y', alpha=0.2, color='#999999')
    plt.xlabel('Frequency Groups')
    plt.savefig('out/sex_g2_freq.png', bbox_inches='tight')
    plt.show()

    sex_g3_freq.plot.bar(stacked=False, color="lightblue")
    plt.title('Sex - Grade 3 Frequencies')
    plt.grid(True, axis='y', alpha=0.2, color='#999999')
    plt.xlabel('Frequency Groups')
    plt.savefig('out/sex_g3_freq.png', bbox_inches='tight')
    plt.show()


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
    # STEP1()
    STEP2()
    # STEP3()
    # STEP4()
