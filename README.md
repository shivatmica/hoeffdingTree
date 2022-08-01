# hoeffdingTree
This is a test implementation for HoeffdingTree classifier implementation from scikit-multiflow. HoeffdingTree is also known as Very Fast Decision Tree (VFDT) is an incremental, anytime decision tree induction algorithm which is capable of learning from massive data streams.

APS Dataset:
https://www.kaggle.com/datasets/uciml/aps-failure-at-scania-trucks-data-set/code?select=aps_failure_test_set.csv
Results:
Hoeffding Tree accuracy: 0.9616 Error Rate: 3.8

Beijing Air Quality Dataset:
https://archive.ics.uci.edu/ml/machine-learning-databases/00501/
Results:
Regressor mean absolute error: 5.5459

HT Classification Model Parameters (APS Dataset):
binary_split': False,
Hoeffding Tree accuracy: 0.9616 
'grace_period': 200,
'leaf_prediction': 'nb',
'max_byte_size': 33554432,
'memory_estimate_period': 1000000,
'nb_threshold': 0,
'no_preprune': False,
'nominal_attributes': None,
'remove_poor_atts': False,
'split_confidence': 1e-07,
'split_criterion': 'info_gain',
'stop_mem_management': False,
'tie_threshold': 0.05

Details of the Tree:
Active leaf byte size estimate: 0.0,
Active learning nodes: 5,
Byte size estimate overhead: 1.0,
Inactive leaf byte size estimate: 0.0,
Tree depth: 3,
Tree size (leaves): 5,
Tree size (nodes): 9


HT Regression Model Parameters (Beijing Air Quality):
{'binary_split': False,
 'grace_period': 200,
 'leaf_prediction': 'perceptron',
 'learning_ratio_const': True,
 'learning_ratio_decay': 0.001,
 'learning_ratio_perceptron': 0.02,
 'max_byte_size': 33554432,
 'memory_estimate_period': 1000000,
 'nb_threshold': 0,
 'no_preprune': False,
 'nominal_attributes': None,
 'random_state': None,
 'remove_poor_atts': False,
 'split_confidence': 1e-07,
 'stop_mem_management': False,
 'tie_threshold': 0.05}

Details of the Tree:
{'Active leaf byte size estimate': 0.0,
 'Active learning nodes': 17,
 'Byte size estimate overhead': 1.0,
 'Inactive leaf byte size estimate': 0.0,
 'Tree depth': 6,
 'Tree size (leaves)': 17,
 'Tree size (nodes)': 33}


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
  X_train: np.array -> The X dataset, which has the data without the label
  y_train: np.array -> The y dataset, which only contains the labels.
  max_samples: int -> The max number of samples the data can contain.
  n_samples: int -> The number of samples the user would like the model to predict and train on.
  classification: bool -> A user preference which tells the code to perform classification 
  regression: bool -> A user preference which tells the code to perform regression.
                      Both classification and regression cannot have the same values. 
  
  Returns: 
    Classification -> Accuracy of the model 
    Regression -> Absolute Mean squared error.
```

# test.py
test.py contains a short 3-line piece of code, on a sample regression dataset. We input the dataset, and call ```init()``` on it. After that we can simply call the model using the ```hoeffding()``` function.



