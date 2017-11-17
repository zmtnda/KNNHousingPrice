
# coding: utf-8

# In[187]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display
from subprocess import check_output
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance




# In[ ]:


def train(X_train, y_train):
    return


# In[ ]:





# In[ ]:





# In[190]:


def predict(X_train, x_test, k):

    distances = []
    targets = []
#     print "str(X_train[0]) " + str(X_train[0]) + ", str(len(x_test)) " + str(len(x_test))

#     for sublist in list:
#         del sublist[index]

    Y_train = np.delete(X_train, 2, axis=1)
    for i in range(len(x_test)):
        # first we compute the euclidean distance
#         print "str(Y_train[i]) " + str(Y_train[i])
        eudistance = distance.euclidean(x_test, Y_train[i])
#         eudistance = distance.euclidean(a, b)
    
#         print "str(X_train[0]) " + str(X_train[0]) + " predict " + str(x_test) + ", distance = " + str(eudistance)
#         distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
        # add it to list of distances
        distances.append([eudistance, i])

    # sort the list
    distances = sorted(distances)

    # make a list of the k neighbors' targets
    for i in range(k):
        salePrices = distances[i][0]
        targets.append(salePrices)

    # return most common target
    return reduce(lambda x, y: x + y, targets) / len(targets)

def kNearestNeighbor(X_train, X_test, predictions, k):
    # check if k larger than n
    
    if k > len(X_train):
        raise ValueError
        
    # train on the input data
#     train(X_train, y_train)

    # loop over all observations
    for i in range(len(X_test)):
#         print str(len(X_train)) + " call predict" + str(k)
        prediction = predict(X_train, X_test[i], k)
        print "X_test[i] " + str(X_test[i]) + ", distance = " + str(prediction)
        predictions.append(prediction)
                                 
#         predictions.append(predict(X_train, X_test[i], k))
        
def main():
    knn = KNeighborsClassifier(n_neighbors=20)
    
    field_train = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', 'TotRmsAbvGrd']
    field_test = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', 'TotRmsAbvGrd']

    train_data = pd.read_csv('./train.csv', skipinitialspace=True, usecols=field_train).astype(str).astype(int)
    Y_train = np.array(train_data['SalePrice']).tolist() 
    display (train_data.head())
    train_data=np.array(train_data).tolist()
    display(train_data[0])
    
    test_data = pd.read_csv('./test.csv', skipinitialspace=True, usecols=field_test).fillna(0).astype(np.int64)
    display (test_data.head())
    test_data=np.array(test_data).tolist()
#     display(test_data)
    display(test_data[0])
    
    # fitting the model
    knn.fit(train_data, Y_train)
    # predict the response
    pred = knn.predict(test_data)
    print knn.score(test_data, pred)
    
    predictions = []
    try:
        kNearestNeighbor(train_data, test_data, predictions, 20)
        predictions = np.asarray(predictions)

        # evaluating accuracy
#         accuracy = accuracy_score(y_test, predictions) * 100
        print('\nThe accuracy of OUR classifier is')

    except ValueError:
        print('Can\'t have more neighbors than training samples!!')

if __name__ == '__main__':
    main()

