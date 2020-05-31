import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA
from random import seed

num_imges_train= 60000
num_imges_valid=10000

def randomforest_run():
    seed(201)
    # load data
    mnist = input_data.read_data_sets("../data/mnist", one_hot=True)
    X_train, Y_train=mnist.train.next_batch(num_imges_train)
    X_test, Y_test=mnist.test.next_batch(num_imges_valid)

    # reshapre
    print(np.shape(X_train))
    print(np.shape(X_test))
    x_train = np.reshape(X_train,(num_imges_train, 28*28))
    y_train=np.reshape(np.argmax(Y_train, 1), (-1, 1))
    x_test = np.reshape(X_test, (num_imges_valid, 28*28))
    y_test=np.reshape(np.argmax(Y_test, 1), (-1, 1))

    # n_extimators: number of iterations; n_jobs: number of CPUs to use
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1).fit(x_train, y_train)
    print(f'Train Accuracy: {rf.score(x_train, y_train)*100}%')
    print(f'Test Accuracy: {rf.score(X_test, y_test)*100}%')

def randomforest_PCA_run():
    seed(201)
    # load data
    mnist = input_data.read_data_sets("../data/mnist", one_hot=True)
    X_train, Y_train=mnist.train.next_batch(num_imges_train)
    X_test, Y_test=mnist.test.next_batch(num_imges_valid)

    # reshapre
    print(np.shape(X_train))
    print(np.shape(X_test))
    x_train = np.reshape(X_train,(num_imges_train, 28*28))
    y_train=np.reshape(np.argmax(Y_train, 1), (-1, 1))
    x_test = np.reshape(X_test, (num_imges_valid, 28*28))
    y_test=np.reshape(np.argmax(Y_test, 1), (-1, 1))

    #compress data with pca
    pca = PCA(n_components=32)
    x_train_pca=pca.fit_transform(x_train)
    x_test_pca=pca.transform(x_test)
    print(np.shape(x_train_pca))
    print(np.shape(x_test_pca))
    print(np.shape(y_train))
    print(np.shape(y_test))
    
    # n_extimators: number of iterations; n_jobs: number of CPUs to use
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1).fit(x_train_pca, y_train)
    print(f'Train Accuracy: {rf.score(x_train_pca, y_train)*100}%')
    print(f'Test Accuracy: {rf.score(x_test_pca, y_test)*100}%')

if __name__ == "__main__":
    #randomforest_run()
    randomforest_PCA_run()
