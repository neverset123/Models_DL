import load_data
import evaluate_split
import randomforest
from random import seed
from math import sqrt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def sonar_run():
    seed(2)
    filePath='../data/sonar.csv'
    dataset = load_data.load_csv(filePath, True)
    # convert string attributes to integers
    for i in range(0, len(dataset[0])-1):
        load_data.str_column_to_float(dataset, i)
    # convert class column to integers
    load_data.str_column_to_int(dataset, len(dataset[0])-1)
    # evaluate algorithm
    n_folds = 5
    max_depth = 10
    min_size = 1
    sample_size = 1.0
    n_features = int(sqrt(len(dataset[0])-1))
    for n_trees in [1, 5, 10]:
        scores = evaluate_split.evaluate_algorithm(dataset, randomforest.random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
        print('Trees: %d' % n_trees)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores)))) 

def mnist_run():
    filePath='../data/mnist'
    num_imges_train= 60000
    num_imges_valid=10000
    seed(201)
    # load data
    mnist = input_data.read_data_sets(filePath, one_hot=True)
    X_train, Y_train=mnist.train.next_batch(num_imges_train)
    X_test, Y_test=mnist.test.next_batch(num_imges_valid)
    y_train=np.reshape(np.argmax(Y_train, axis=1),(-1, 1))
    y_test=np.reshape(np.argmax(Y_test, axis=1),(-1,1))
    print(np.shape(X_train))
    print(np.shape(y_train))
    dataset_train=np.concatenate((X_train, y_train), axis=1)
    dataset_test=np.concatenate((X_test, y_test), axis=1)
    dataset=np.concatenate((dataset_train, dataset_test), axis=0)

    # shape
    print(np.shape(dataset_train))
    print(np.shape(dataset_test))
    print(np.shape(dataset))
    # evaluate algorithm
    n_folds = 10
    max_depth = 20
    min_size = 1
    sample_size = 1.0 # 1 means no subsampling, all datas are taken
    n_features = 28*28
    n_trees=100
    scores = evaluate_split.evaluate_algorithm(dataset, randomforest.random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
    print('Trees: %d' % n_trees)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores)))) 



if __name__ == "__main__":
    mnist_run()