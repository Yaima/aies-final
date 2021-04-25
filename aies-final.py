import random

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier

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

    df.loc[df['age'].between(15, 17, inclusive=True), ['age_group']] = '15-17'
    df.loc[df['age'].between(18, 22, inclusive=True), ['age_group']] = '18-22'

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

    age_unique.sort()
    g1_unique.sort()
    g2_unique.sort()
    g3_unique.sort()

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

    bins = [0, 11, 19]
    labels = [0, 1]

    df['G1'] = pd.cut(df.G1, bins=bins, labels=labels, include_lowest=True)
    df['G2'] = pd.cut(df.G2, bins=bins, labels=labels, include_lowest=True)
    df['G3'] = pd.cut(df.G3, bins=bins, labels=labels, include_lowest=True)

    age_g1_freq = df.groupby(['age_group', 'G1']).size()
    age_g1_freq.to_csv('out/age_g1_freq.csv')

    age_g2_freq = df.groupby(['age_group', 'G2']).size()
    age_g2_freq.to_csv('out/age_g2_freq.csv')

    age_g3_freq = df.groupby(['age_group', 'G3']).size()
    age_g3_freq.to_csv('out/age_g3_freq.csv')

    sex_g1_freq = df.groupby(['sex', 'G1']).size()
    sex_g1_freq.to_csv('out/sex_g1_freq.csv')

    sex_g2_freq = df.groupby(['sex', 'G2']).size()
    sex_g2_freq.to_csv('out/sex_g2_freq.csv')

    sex_g3_freq = df.groupby(['sex', 'G3']).size()
    sex_g3_freq.to_csv('out/sex_g3_freq.csv')

    age_g1_freq.plot.bar(stacked=False, color="salmon")
    plt.title('Age - First Period Grade')
    plt.grid(True, axis='y', alpha=0.2, color='#999999')
    plt.xlabel('Age - G1 Discretized')
    plt.savefig('out/age_g1_freq.png', bbox_inches='tight')
    plt.show()

    age_g2_freq.plot.bar(stacked=False, color="salmon")
    plt.title('Age - Second Period Grade')
    plt.grid(True, axis='y', alpha=0.2, color='#999999')
    plt.xlabel('Age - G2 Discretized')
    plt.savefig('out/age_g2_freq.png', bbox_inches='tight')
    plt.show()

    age_g3_freq.plot.bar(stacked=False, color="salmon")
    plt.title('Age - Final Grade')
    plt.grid(True, axis='y', alpha=0.2, color='#999999')
    plt.xlabel('Age - G3 Discretized')
    plt.savefig('out/age_g3_freq.png', bbox_inches='tight')
    plt.show()

    sex_g1_freq.plot.bar(stacked=False, color="lightblue")
    plt.title('Sex - First Period Grade')
    plt.grid(True, axis='y', alpha=0.2, color='#999999')
    plt.xlabel('Sex - G1 Discretized')
    plt.savefig('out/sex_g1_freq.png', bbox_inches='tight')
    plt.show()

    sex_g2_freq.plot.bar(stacked=False, color="lightblue")
    plt.title('Sex - Second Period Grade')
    plt.grid(True, axis='y', alpha=0.2, color='#999999')
    plt.xlabel('Sex - G2 Discretized')
    plt.savefig('out/sex_g2_freq.png', bbox_inches='tight')
    plt.show()

    sex_g3_freq.plot.bar(stacked=False, color="lightblue")
    plt.title('Sex - Final Grade')
    plt.grid(True, axis='y', alpha=0.2, color='#999999')
    plt.xlabel('Sex - G3 Discretized')
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
    #school,sex,age,address,famsize,Pstatus,Medu,Fedu,Mjob,Fjob,reason,guardian,traveltime,studytime,failures,schoolsup,famsup,paid,activities,nursery,higher,internet,romantic,famrel,freetime,goout,Dalc,Walc,health,absences,G1,G2,G3

    raw_df = pd.read_csv(r"dataset/student-por.csv", delimiter=',')
    df = raw_df.replace({'higher': {'yes': True, 'no': False}})

    # print("Number of records: ")
    # r, c = df.shape
    # print("rows: ", r)
    # print("columns: ", c)
    # print()

    num_rows = int(len(df))

    # split data into train and test sets
    shuffled = df.sample(frac=1)
    df_train = shuffled.iloc[:int(num_rows / 2)]
    df_test = shuffled.iloc[int(num_rows / 2):]


    # classify data Y in this case is G3. Features are G1 and G2,

    features = ['failures', 'Medu', 'studytime', 'absences', 'G1', 'G2', 'higher']
    y_feature = 'G3'

    # create training data
    df_train_x = df_train[features]
    df_train_y = df_train[y_feature]

    # create and train classifier
    orig_classifier = DecisionTreeClassifier()
    orig_classifier.fit(df_train_x, df_train_y)

    # run classifier against the test set
    df_test_x = df_test[features]
    predicted = orig_classifier.predict(df_test_x)

    df_test_x['Predicted Grade'] = predicted
    df_test_x['Actual Grade'] = df_test[y_feature]

    # TODO create a classifier for the data from Step 3.3

    print(df_test_x.head())


if __name__ == '__main__':
    # STEP1()
    # STEP2()
    # STEP3()
    STEP4()
