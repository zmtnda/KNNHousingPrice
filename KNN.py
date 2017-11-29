
# coding: utf-8

# In[71]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display
from subprocess import check_output
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance



# In[72]:


def train(X_train, y_train):
    return


# In[73]:


def kNearestNeighbor(X_train, X_test, predictions, k):
    # check if k larger than n
    
    if k > len(X_train):
        raise ValueError
        
    # train on the input data
    # train(X_train, y_train)

    # loop over all observations
    for i in range(len(X_test)):
        prediction = predict(X_train, X_test[i], k)
        predictions.append(prediction)
                                 
def predict(X_train, x_test, k):

    distances = []
    targets = []

    Y_train = np.delete(X_train, 10, axis=1)
    for i in range(len(Y_train)):
        # first we compute the euclidean distance
        eudistance = distance.euclidean(x_test, Y_train[i])

        # add it to list of distances
        distances.append([eudistance, i])

    # sort the list
    distances = sorted(distances)

    # make a list of the k neighbors' targets
    for i in range(k): 
        index = distances[i][1]
        salePrices = X_train[index][10]
        targets.append(salePrices)

    # return the mean
    return reduce(lambda x, y: x + y, targets) / len(targets)
        
    
def calculate_distance(my_knn, sklearn_knn):
    sum = 0
    count = 0
    difference = 0;
#     print str(len(my_knn))
    for i in range(len(my_knn)):
        temp = abs(my_knn[i] - sklearn_knn[i])
        if temp == 0:
            count += 1
        elif difference < temp:
            difference = temp
            print str(my_knn[i]) + " - " + str(sklearn_knn[i]) + "=" + str(difference) 
     
    return count * 100.00/len(my_knn) 
    
def main():
    
    field_train = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', 'TotRmsAbvGrd']
    field_test = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', 'TotRmsAbvGrd']

    train_data = pd.read_csv('./train.csv', skipinitialspace=True, usecols=field_train).astype(str).astype(int)
    target_col = np.array(train_data['SalePrice']).tolist() 
    display (train_data.head())
    train_data=np.array(train_data).tolist()
    display(train_data[0])
    
    test_data = pd.read_csv('./test.csv', skipinitialspace=True, usecols=field_test).fillna(0).astype(np.int64)
    display (test_data.head())
    test_data=np.array(test_data).tolist()
    display(test_data[0])
    
    # sklearn.neighbors' KNN model
    knn = KNeighborsClassifier(n_neighbors=1)
    # delete the 10th column 'SalePrice'
    Y_train = np.delete(train_data, 10, axis=1)
    # fitting the model
    knn.fit(Y_train, target_col)
    # predict the response
    sklearn_predictions = knn.predict(test_data)

    # Our KNN model
    our_predictions = []
    try:
        kNearestNeighbor(train_data, test_data, our_predictions, 1)
        our_predictions = np.asarray(our_predictions)

        # evaluating accuracy
        print "sklearn' KNN model's output: " +str(sklearn_predictions) + " " + str(len(sklearn_predictions))
        print "Our' KNN model's output:     " +str(our_predictions) + " " + str(len(our_predictions))
        print('\nThe accuracy of OUR classifier is ' + "%.2f" % round(calculate_distance(our_predictions, sklearn_predictions),2) + '%')

    except ValueError:
        print('Can\'t have more neighbors than training samples!!')

if __name__ == '__main__':
    main()

