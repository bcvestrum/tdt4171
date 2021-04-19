import pandas as pd 
import math
from graphviz import Digraph

class Decision_tree: #create a decision tree for the question
    def __init__(self, attribute):
        self.attribute = attribute
        self.sub_trees = {} #children

    def __str__(self):
        return self.attribute

    def add_sub_trees(self, sub_tree, value):
        self.sub_trees[value] = sub_tree
    
    def predict(self, example):
        value = example[self.attribute]
        if isinstance(self.sub_trees[value], Decision_tree):
            return self.sub_trees[value].predict(example)
        else:
            return self.sub_trees[value]
    

def plurality_value_function(examples):
    positive = 0
    negative = 0
    for _, row in examples.iterrows():
        if row["Survived"] == 1:
            positive += 1
        else:
            negative += 1
    return 1 if positive >= negative else 0

def same_value_function(examples):
    expected_value = examples.sum(axis=0, skipna=True)["Survived"]/len(examples)
    return int(expected_value) == expected_value

def decision_tree_learning(examples, attributes, parent_examples, index ):
    if len(examples) == 0: #check if examples is empty
        return plurality_value_function(parent_examples) #return plurality value of parent_examples
    if same_value_function(examples):
        return examples.iloc[0]["Survived"]
    if len(attributes) == 0:
        return plurality_value_function(examples)

    #else
    argmax_attribute = information_gain_function(examples, attributes)
    tree = Decision_tree(argmax_attribute)

    for value in unique_value_function(examples, argmax_attribute):
        exs = examples.loc[examples[argmax_attribute] == value]
        new_attributes = attributes.copy()
        new_attributes.remove(argmax_attribute)
        sub_trees = decision_tree_learning(exs, new_attributes, examples, index + 1)
        tree.add_sub_trees(sub_trees, value)
    return tree

def read_data(filename):
    csvfile = pd.read_csv(filename, header=0)
    return csvfile

def b_function(value):
    if value >= 1 or value <= 0:
        return 0 
    result = -(value*math.log(value, 2) + (1-value)*math.log(1-value,2)) #function from textbook
    return result

def information_gain_function(examples, attributes):#function for information gain
    positive_cases = 0
    negative_cases = 0

    for index, row in examples.iterrows():
        if row["Survived"] == 1:
            positive_cases += 1
        else:
            negative_cases += 1

    max_information_gain_value = -math.inf
    argmax_attribute = ""
    for attribute in attributes:
        local_gain = b_function(positive_cases/(positive_cases + negative_cases)) - remainder_function(examples, attribute, positive_cases, negative_cases)
        if local_gain >= max_information_gain_value:
            max_information_gain_value = local_gain
            argmax_attribute = attribute
    
    return argmax_attribute

def remainder_function(examples, attribute, positive_cases, negative_cases):
    remainder = 0
    for value in unique_value_function(examples, attribute):
        positive_k_cases = 0
        negative_k_cases = 0

        for index, row in examples.iterrows():
            if row[attribute] == value:
                positive_k_cases += 1
            else:
                negative_k_cases += 1
        
        remainder += (positive_k_cases + negative_k_cases)/(positive_cases + negative_cases) * b_function(positive_k_cases/(positive_k_cases + negative_k_cases))
    return remainder

def unique_value_function(examples, attribute):
    return list(set(examples[attribute].values))

def accuracy_function(dataset, tree):
    right_answer = 0

    for row in dataset.iloc:
        if tree.predict(row) == row["Survived"]:
            right_answer += 1
    return right_answer / len(dataset)

def insert_split_point_into_dataset(dataset, attribute, split_point):
    for index, row in dataset.iterrows():
        if row[attribute] > split_point:
            dataset.at[index, attribute] = split_point
        else:
            dataset.at[index, attribute] = 0

def find_optimal_splitting_point(training_data):
    average_fare = training_data.sum(axis=0, skipna=True)["Fare"]/len(training_data)
    return average_fare
        
def main():
    training_data = read_data("./train.csv")
    test_data = read_data("./test.csv")

    trained_tree = decision_tree_learning(training_data, ["Sex"], [], 0)
    print(accuracy_function(test_data, trained_tree))

    optimal_splitting_point = find_optimal_splitting_point(training_data)
    insert_split_point_into_dataset(training_data, "Fare", optimal_splitting_point)
    insert_split_point_into_dataset(test_data, "Fare", optimal_splitting_point)
    trained_tree_with_continous = decision_tree_learning(training_data, ["Sex", "Fare"], [], 0)
    print(accuracy_function(test_data, trained_tree_with_continous))



if __name__ == '__main__':
    main()
    """
    dot = Digraph(comment='Decision tree')

    dot.node('sex1', 'Sex')
    dot.node('sex2', 'Sex')
    dot.node('sur1', 'Survived')
    dot.node('not_sur1', 'Not survived')
    dot.node('sur2', 'Survived')
    dot.node('not_sur2', 'Not survived')
    dot.node("over", 'Fare')

    dot.edge('over', 'sex1', '< average_fare')
    dot.edge('over', 'sex2', '> average_fare')
    dot.edge('sex1', 'sur1', 'female')
    dot.edge('sex1', 'not_sur1', 'male')
    dot.edge('sex2', 'sur2', 'female')
    dot.edge('sex2', 'not_sur2', 'male')


    dot.render('taskb.gv', view=True)
    """
