from extract import extract_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from math import sqrt
from node import Node
from random import *
from math import *


'''
For the Thoracic Surgery classification, I will be using neural random forest. 

Random Forest is a supervised learning that consists of neural decision trees. 
Decision trres are created by randomly selecting sample data. The final 
prediction is calculated through majority vote. 

Base Random Forest Classifier Accuracy = 81% 

Author: Ly Sung
Date: March 21st 2019
'''


'''
Important parameters:
1. input
2. output
3. number of trees
4. bagging approach, sqr/log2
5. input size
6. depth of the decision tree - 
'''
class NeuralRandomForest():
    def __init__(self, depth = 10, min_leaf=10, num_trees = 10):
        data, target = extract_data()
        
        # splits the data into training and testing 
        X_train, X_test, Y_train, Y_test = \
            train_test_split(data, target, test_size=0.3)

        self.max_depth = depth
        self.min_leaf = min_leaf
        self.n_features = int(sqrt(len(data[0])-1))
        self.num_trees = num_trees

        self.combine_lists(X_train, Y_train)
        self.combine_lists(X_test, Y_test)
        # a = self.split_point(X_train, Y_train, self.feature_size)

        # print(a)
        # self.generate_decision_trees(X_train, Y_train)
        a = self.generate_random_forest(X_train, Y_train, X_test, Y_test)


    def combine_lists(self, lista, listb):
        for i in range(len(lista)):
            lista[i].append(listb[i])

    '''
    Selects a random number of samples from the original set are use to construct
    the decision tree. The idea is to the decrease the correlation among all
    the neural trees. 

    At each point, only n number of features are considered. We will be using
    the sqrt root(total number of features)

    @NOTE: has replacement 
    '''
    def bagging(self, data_set, data_target, ratio = 1):
        selected_samples = []
        selected_samples_targets = []
        data_set_size = len(data_set)

        total_sample = round(data_set_size * ratio)
        
        while len(selected_samples) < total_sample:
            index = randrange(data_set_size)
            selected_samples.append(data_set[index])
            selected_samples_targets.append(data_set[index][-1])
        
        return selected_samples, selected_samples_targets


    '''
    Splits a given data set into left and right sides. 
    @NOTE: data_set = [[], []]
    '''
    def split_dataset(self, data_set, index, value):
        left_set = []
        right_set = []

        for data in data_set:
            if data[index] < value:
                left_set.append(data)
            else:
                right_set.append(data)

        return left_set, right_set

    '''
    The gini index calculates how often a randomly selected set is incorrectly
    labelled. 
    
    @NOTE: cost function = 1 - sum of (proportion)^2
    - perefectly classified gini index = 0 
    '''
    def cal_gini_score(self, groups, target_set):
        gini_sum = 0

        for target in target_set:
            for data in groups: # data : left / right 
                if len(data) <= 0:
                    continue
                
                data_target = []
                group_size = len(data)

                for i in range(group_size):
                    data_target.append(data[-1])

                proportion = (data_target.count(target)) / float(group_size)
                gini_sum += proportion * (1 - proportion)

        return gini_sum
    
    '''
    To find the best split within a tree involves calculating the cost function
    of each value. For this we will be using the Gini Index where 0 represents
    the perfect split since there are only two classes. 

    @NOTE: data_set & target_set is a sample of the original training data_set 
    n_features - is a fixed number of features choosen

    @TODO: move to bagging def
    '''
    def split_point(self, data_set, target_set):
        # randomly select n features from the data_set 
        selected_features = []


        input_size = len(data_set[0])
        while len(selected_features) < self.n_features:
            index = randrange(input_size) - 1
            if index not in selected_features:
                selected_features.append(index)

        min_gini_value = 9999
        best_col_index = 9999
        best_cutoff_val = 9999
        best_groups = None

        # calculate minimum gini index 
        for index in selected_features:
            for data in data_set:
                data_value = data[index]

                groups = self.split_dataset(data_set, index, data_value)
                gini = self.cal_gini_score(groups, target_set)

                if gini < min_gini_value:
                    min_gini_value = gini
                    best_cutoff_val = data_value
                    best_col_index = index
                    best_groups = groups
        
        return {
            "index": best_col_index, #index where gini index is minimized 
            "value": best_cutoff_val,
            "groups": best_groups
        }

    def get_targets(self, data_set):
        targets = [data[-1] for data in data_set]
        return targets

    '''
    Determines the highest frequency feature - node value
    '''
    def freq_output(self, target_set):
        return max(set(target_set), key=target_set.count)


    def generate_node(self, node, split_section, depth):
        left_group, right_group = split_section["groups"]

        left_group_target = self.get_targets(left_group)
        right_group_target = self.get_targets(right_group)
        target_set = self.get_targets(left_group + right_group)

        # return if node is None
        if not node.left and not node.right:
            return

        # if no split is required | ie. leaf nodes
        if len(left_group) <= 0 or len(right_group) <= 0:
            node_value = self.freq_output(target_set)
            # print("here")
            # print(node.value)
            node.left.set_value(node_value)
            node.right.set_value(node_value)
            return

        # if max-depth is reached  
        if depth >= self.max_depth:
            node.left.set_value(self.freq_output(left_group_target))
            node.right.set_value(self.freq_output(right_group_target))
            return

        # left child
        if len(left_group) < self.min_leaf:
            node.left.set_value(self.freq_output(left_group_target))
        else:
            split_section = self.split_point(left_group, left_group_target)

            new_node = Node()
            new_node.set_value(split_section["value"])
            new_node.set_index(split_section["index"])
            new_node.parent = node
            left_node = Node(node, None, None, round(random(), 2))
            right_node = Node(node, None, None, round(random(), 2))

            self.generate_node(new_node, split_section, depth + 1)
            node.set_left_node(new_node)

        # right child
        if len(right_group) < self.min_leaf:
            node.right.set_value(self.freq_output(right_group_target))
        else:
            split_section = self.split_point(right_group, right_group_target)

            new_node = Node()
            new_node.set_value(split_section["value"])
            new_node.set_index(split_section["index"])
            new_node.parent = node
            left_node = Node(node, None, None, round(random(), 2))
            right_node = Node(node, None, None, round(random(), 2))

            self.generate_node(new_node, split_section, depth + 1)
            node.set_right_node(new_node)

    '''
    Initializes neural decision trees. 
    From each node to another, there will be an associataed weight. 
    '''
    def generate_decision_trees(self, training_set, target_set):
        print("Generating decision tree")

        current_depth = 1 
        split_section = self.split_point(training_set, target_set)
        
        # initializes nodes
        root = Node()
        root.set_value(split_section["value"])

        
        root.set_index(split_section["index"])
        left_node = Node(root, None, None, round(random(), 2))
        right_node = Node(root, None, None, round(random(), 2))

        root.set_left_node(left_node)
        root.set_right_node(right_node)
        
        self.generate_node(root, split_section, current_depth)

        print("Finished generating decision tree")

        return root

    '''
    Determines what the output actually is. The sigmoid function takes in 
    an input and produces an output between 0 and 1. It is also great for 
    calculating the slope for backpropagating error.

    @TODO: convert result to whole number
    '''
    def sigmoid(self, activation):
        try:
            return 1.0 / (1.0 + exp(-activation))
        except OverflowError:
            activation = activation /1000
            return 1 / (1 + exp(-activation))
    
    # --- BACKPROP ----
    '''
    Calculates the slope of the output of a neuron
    where y = output
    '''
    def sigmoid_prime(self, y):
        return y * (1.0 - y)

    def sum_error(self, errors, output):
        return errors + self.sigmoid_prime(output)
    '''
    Calculates the error of each neuron. 
    error = (d-y) * y * (1-y)

    Once the error is calculated, prop. the error back all the way back to
    input layer. 
    '''
    def back_error_prop(self, output, expected_output):
        error = (expected_output - output)

        error = self.sum_error(error, output)
        return error 
    # --- BACKPROP ----

    '''
    Predicts the result from the neural decision tree. 
    '''
    def predict(self, tree, row):
        gini_index = tree.get_index()
        
        value_at_gini = row[gini_index]

        if value_at_gini < tree.get_value():
            if tree.has_left() and tree.left.has_left():
                return self.predict(tree.left, row)
            else:
                activation = tree.value + (value_at_gini - tree.weight)
                output = round(self.sigmoid(activation), 2)
                # error prob here 
                sum_error = self.back_error_prop(output, row[-1])
                tree.update_weight(sum_error, row[gini_index])
                return output
            
        else:
            if tree.has_right() and tree.right.has_left():
                return self.predict(tree.right, row)
            else:
                activation = tree.value + (value_at_gini - tree.weight)
                output = round(self.sigmoid(activation), 2)
                # error prob here 
                sum_error = self.back_error_prop(output, row[-1])
                tree.update_weight(sum_error, row[gini_index])
                return output
    
    '''
    Predict the result by averaging all the neural decision tree predictions.
    '''
    def final_predict(self, forest, data):
        predictions = []

        for tree in forest:
            tree_result = self.predict(tree, data)
            predictions.append(tree_result)

        return max(set(predictions), key=predictions.count)

    def data_rescale(self, data):
        new_set = []
        average = sum(data) / len(data)
        ones = 0
        zeros = 0

        print("Data: ")
        print(data)
        print("Average: ", average)


        for result in data:
            if result < average * 1.1:
                new_set.append(0)
                zeros += 1
            else:
                new_set.append(1)
                ones += 1

        return new_set, ones, zeros

    '''
    calculates accuracy
    '''
    def accuracy(self, outputs, targets):
        wrong = 0
        total_length = len(outputs)
        total_accuracy = 0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for i in range(len(outputs)):
            if outputs[i] != targets[i]:
                wrong += 1

        total_accuracy = (total_length-wrong) / total_length

        return total_accuracy
    '''
    Generate random forest
    '''
    def generate_random_forest(self, data_set, data_target, testing_data, testing_target):
        print("Generating Random forest...")
        forest = []

        for i in range(self.num_trees):
            random_sample, random_sample_targets = self.bagging(data_set, data_target)
            neural_tree = self.generate_decision_trees(random_sample, random_sample_targets)
            forest.append(neural_tree)
        
        predictions = []
        for data in testing_data:
            predictions.append(self.final_predict(forest, data))

        final_results, success, failure = self.data_rescale(predictions)
        print("Actual")
        print(testing_target)


        print(final_results)
        print("True: ", success)
        print("False: ", failure)

        acurracy_result = self.accuracy(final_results, testing_target)
        print("Total accuracy: ", acurracy_result)


NeuralRandomForest()