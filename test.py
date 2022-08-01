from utils import *

if __name__ == "__main__":
    # regression
    X_train, y_train, X_test, y_test = init(
        label="PM2.5",
        train="C:\\Users\\Asus\\Downloads\\PRSA_Data_Dingling_20130301-20170228.csv",
        split=True,
    )
    print(
        hoeffding(X_test, y_test, regression=True, max_samples=100000, n_samples=10000)
    )
