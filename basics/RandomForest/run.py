import load_data
import evaluate_split
import randomforest
from random import seed
from math import sqrt

filePath='./sonar.csv'

def run():
    seed(2)
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

if __name__ == "__main__":
    run()