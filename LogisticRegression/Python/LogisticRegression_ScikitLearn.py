from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np

#Loading the Iris Dataset
dataset = datasets.load_iris()

# Printing infos
print (dataset['feature_names'], dataset['target_names'])

X_training = dataset['data']
y_training = (dataset['target'] == 2).astype(np.int) # 1 if Iris-Virginica, else 0


logistic_regressor = LogisticRegression()
# Training the model
logistic_regressor.fit(X_training,y_training)


# Testing the model - generating some random flower
X_testing = np.array([np.random.uniform(0.1,6), np.random.uniform(2.5,4), np.random.uniform(1.,6), np.random.uniform(0.2,3)]).reshape(1,-1)
y_predict = logistic_regressor.predict(X_testing)

# Is this example an Iris Virginica? 1 yes (positive class) , 0 no (negative class)
print("Testing example {}".format(X_testing))

if y_predict == 1:
    print("Example is Iris-Virginica: y={}".format(y_predict))
else:
    print("Example is NOT Iris Virginica: y={}".format(y_predict))