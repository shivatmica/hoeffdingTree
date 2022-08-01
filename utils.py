import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skmultiflow.data import SEAGenerator
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.data import RegressionGenerator
from skmultiflow.trees import HoeffdingTreeRegressor
from sklearn.model_selection import train_test_split


def init(
    label,
    test_size=0.2,
    train_size=0.8,
    train="aps_failure_training_set_processed_8bit.csv",
    test="aps_failure_test_set_processed_8bit.csv",
    split=False,
):
    if test_size + train_size != 1.0:
        raise Exception(
            f"test_size and train_size must add up to 1.0, but got {train_size + test_size} instead."
        )

    if split:
        main_df = pd.read_csv(train)
        if label not in main_df.columns:
            raise Exception(f"{label} not found in the DataFrame.")
        X = main_df.drop(label, axis=1)
        y = main_df[label]

        X_train, y_train, X_test, y_test = train_test_split(
            X, y, train_size=train_size, test_size=test_size, random_state=101
        )

    else:
        train_df = pd.read_csv(train)
        test_df = pd.read_csv(test)
        if label not in train_df.columns:
            raise Exception(f"{label} not found in the train dataset.")
        if label not in test_df.columns:
            raise Exception(f"{label} not found in the test dataset.")

        X_train = train_df.drop(label, axis=1)
        y_train = train_df[label]
        X_test = test_df.drop(label, axis=1)
        y_test = test_df[label]

    return (X_train, y_train, X_test, y_test)


def hoeffding(
    X_train, y_train, max_samples, n_samples, classification=False, regression=False
):
    if classification == False and regression == False:
        raise Exception(
            "No mode selected, please set classification or regression to true."
        )
    if classification == True and regression == True:
        raise Exception("Both parameters are True, please set only one as True.")

    if classification:
        stream = SEAGenerator(random_state=1)
        ht = HoeffdingTreeClassifier(leaf_prediction="nb")

        curr_samples = 0
        tp = 0
        fp = 0
        max_samples = max_samples

        while curr_samples < max_samples and stream.has_more_samples():
            X_test, y_test = stream.next_sample()
            y_pred = ht.predict(X_test)
            if y_test[0] == y_pred[0]:
                tp += 1
            else:
                fp += 1
            ht = ht.partial_fit(X_test, y_test)
            curr_samples += 1

        return {tp / curr_samples}

    if regression:
        stream = RegressionGenerator(random_state=1, n_samples=10000)

        ht_reg = HoeffdingTreeRegressor()

        n_samples_cnt = 0
        max_samples = 100000
        y_pred = np.zeros(max_samples)
        y_true = np.zeros(max_samples)

        while n_samples_cnt < max_samples and stream.has_more_samples():
            X_train, y_train = stream.next_sample()
            y_true[n_samples_cnt] = y_train[0]
            y_pred[n_samples_cnt] = ht_reg.predict(X_train)[0]
            ht_reg.partial_fit(X_train, y_train)
            n_samples_cnt += 1

        return np.mean(np.abs(y_true - y_pred))
