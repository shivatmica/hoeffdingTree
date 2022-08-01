# hoeffdingTree

# utils.py
utils.py containes some inline functions, so that a user can use the hoeffdingTreeClassifier or Regressor to do their own tasks, without performing all the steps of splitting data or actually implementing the model. The user only has to enter a number of parameters which include details of the user's preferences. Through those parameters, the functions return a single value.

If the problem is classification, it would return the accuracy, but if it is regression, the function would return the mean absolute error.

Functions:
```
init(label, test_size = 0.2, train_size = 0.8, train = 'aps_failure_training_set_processed_8bit.csv', test = 'aps_failure_test_set_processed_8bit.csv', split = False):
  label: str -> the label that the classifier has to predict
  test_size: float -> If split is true, we can use the user preferred test rates in train_test_split()
  train_size: float -> If split is true, we can use the user preferred train rates in train_test_split()
  train: str -> The train input dataset from the user. If there is only one available dataset, the train paths will be used.
  test: str -> The test input dataset from the user, if split is False.
  split: bool -> If split is true, only one dataFrame will be read (train), and train_test_split() will be performed.
                 If split is false, two dataFrames will be used (train & test), train_test_split() will not be performed.
  
  Returns:
    (X_train, y_train, X_test, y_test) 
   Four numpy arrays which contain all the data to input in the model.
```

```
hoeffding(X_train, y_train, max_samples, n_samples, classification = False, regression = False):
  X_train: np.array -> 

# test.py
test.py contains a short 3-line piece of code, on a sample regression dataset. We input the dataset, and call ```init()``` on it. 
