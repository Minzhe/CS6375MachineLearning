#########################################################
######              Decision Tree                   #####
#########################################################

import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description='Decision Tree.')
parser.add_argument('-r', '--training', help='Path of training dataset')
parser.add_argument('-v', '--validation', help='Path of validation dataset')
parser.add_argument('-t', '--testing', help='Path of test dataset')
parser.add_argument('-p', '--pruning', help='Pruning factor')
args = parser.parse_args()
train_path = args.training
validation_path = args.validation
test_path = args.testing
pruning_factor = float(args.pruning)

train_data = pd.read_csv(train_path)
validation_data = pd.read_csv(validation_path)
test_data = pd.read_csv(test_path)

class DecisionTree(object):
    def __init__(self, data):
        self.data = data
        self.tree = buildTree(data)

    def printTree(self, prune=False):
        if prune == False:
            headNode = self.tree[0]
        else:
            headNode = self.prunedTree[0]
        headNode.left.recursivePrintNodes()
        headNode.right.recursivePrintNodes()

    def predict(self, newdata, prune=False):
        prediction = list()
        for idx in newdata.index.values:
            entry = newdata.loc[idx]
            if prune == False:
                prediction.append(predictOneRow(self.tree, entry))
            else:
                prediction.append(predictOneRow(self.prunedTree, entry))
        return prediction

    def prune(self, factor):
        n = round(factor * len(self.tree))
        self.prunedTree = self.tree
        for i in range(n):
            self.prunedTree = pruneOneNode(self.prunedTree)

    def preAccuracy(self, val_data, test_data):
        print('\nPre-Pruned Accuracy\n-------------------------')
        print('Number of training instances = {}'.format(len(self.data.index)))
        print('Number of training attributes = {}'.format(len(list(self.data.columns)[:-1])))
        print('Total number of nodes in the tree = {}'.format(len(self.tree)))
        leafNodes = [node for node in self.tree if node.property == 'leafNode']
        print('Number of leaf nodes in the tree = {}'.format(len(leafNodes)))
        truth = self.data.Class
        pred_train = self.predict(self.data)
        pred_val = self.predict(val_data)
        pred_test = self.predict(test_data)
        acc_train = round(100*cal_accuracy(truth, pred_train), 1)
        acc_val = round(100*cal_accuracy(truth, pred_val), 1)
        acc_test = round(100*cal_accuracy(truth, pred_test), 1)
        print('Accuracy of the model on the training dataset = {}%\n'.format(acc_train))
        print('Number of validation instances = {}'.format(len(val_data.index)))
        print('Number of validation attributes = {}'.format(len(list(val_data.columns)[:-1])))
        print('Accuracy of the model on the validation dataset before pruning = {}%\n'.format(acc_val))
        print('Number of testing instances = {}'.format(len(test_data.index)))
        print('Number of validation attributes = {}'.format(len(list(test_data.columns)[:-1])))
        print('Accuracy of the model on the testing dataset = {}%\n'.format(acc_test))

    def postAccuracy(self, val_data, test_data):
        print('\nPost-Pruned Accuracy\n-------------------------')
        print('Number of training instances = {}'.format(len(self.data.index)))
        print('Number of training attributes = {}'.format(len(list(self.data.columns)[:-1])))
        print('Total number of nodes in the tree = {}'.format(len(self.prunedTree)))
        leafNodes = [node for node in self.prunedTree if node.property == 'leafNode']
        print('Number of leaf nodes in the tree = {}'.format(len(leafNodes)))
        truth = self.data.Class
        pred_train = self.predict(self.data, prune=True)
        pred_val = self.predict(val_data, prune=True)
        pred_test = self.predict(test_data, prune=True)
        acc_train = round(100*cal_accuracy(truth, pred_train), 1)
        acc_val = round(100*cal_accuracy(truth, pred_val), 1)
        acc_test = round(100*cal_accuracy(truth, pred_test), 1)
        print('Accuracy of the model on the training dataset = {}%\n'.format(acc_train))
        print('Number of validation instances = {}'.format(len(val_data.index)))
        print('Number of validation attributes = {}'.format(len(list(val_data.columns)[:-1])))
        print('Accuracy of the model on the validation dataset after pruning = {}%\n'.format(acc_val))
        print('Number of testing instances = {}'.format(len(test_data.index)))
        print('Number of validation attributes = {}'.format(len(list(test_data.columns)[:-1])))
        print('Accuracy of the model on the testing dataset = {}%\n'.format(acc_test))


class Node(object):
    def __init__(self, data, name=None, label=None, depth=None):
        self.data = data
        self.entropy = cal_entropy(self.data['Class'])
        self.name = name
        self.label = label
        self.depth = depth
        if self.entropy == 0:
            self.property = 'leafNode'
            self.makeTerminal()
        else:
            self.property = 'innerNode'

    def nodeInfo(self):
        print('Name: ', self.name, sep='')
        print('Entropy: ', self.entropy, sep='')
        print('Label: ', self.label, sep='')
        print('Depth: ', self.depth, sep='')
        print('Property: ', self.property, '\n', sep='')

    def singleSplitNode(self, param):
        self.left = Node(self.data.loc[self.data[param] == 0], name=self.name+'-{}0'.format(param))
        self.right = Node(self.data.loc[self.data[param] > 0], name=self.name+'-{}1'.format(param))
        self.infoGain = cal_infoGain(self, self.left, self.right)
        self.left.property = self.right.property = 'leafNode'
        self.property = 'innerNode'

    def optimalSplitNode(self, param_list):
        tempNode = Node(self.data, name=self.name)
        tempInfoGain = -1
        for param in param_list:
            tempNode.singleSplitNode(param)
            # print(param, ': ', tempNode.infoGain, sep='')
            if tempNode.left.label is not None and tempNode.right.label is not None:
                optimalParam = param
                break
            elif tempNode.infoGain > tempInfoGain:
                optimalParam = param
                tempInfoGain = tempNode.infoGain
        self.param = optimalParam
        self.singleSplitNode(optimalParam)
        # print('Split param: ', self.param)
        # print('InfoGain: ', self.infoGain)

    def makeTerminal(self):
        class_freq = self.data.Class
        class0_freq = sum(class_freq == 0)
        class1_freq = sum(class_freq == 1)
        self.property == 'leafNode'
        if class1_freq > class0_freq:
            self.label = 1
        elif class1_freq < class0_freq:
            self.label = 0
        else:
            self.label = int(np.random.choice(2, 1))

    def printNode(self):
        attr = self.name.split('-')[-1].strip('0').strip('1')
        value = self.name.split('-')[-1][-1]
        label = self.label if self.label is not None else ''
        print('| '*(self.depth-1), attr, ' = ', value, ': ', label, sep = '')

    def recursivePrintNodes(self):
        self.printNode()
        if self.left.label is None:
            self.left.recursivePrintNodes()
        else:
            self.left.printNode()
        if self.right.label is None:
            self.right.recursivePrintNodes()
        else:
            self.right.printNode()


def cal_entropy(num_list):
    if len(num_list) > 0:
        p = sum(num_list) / len(num_list)
        if p == 0:
            return 0
        else:
            return -p*np.log2(p)-(1-p)*np.log2(1-p)
    else:
        return 0

def cal_infoGain(node, leftNode, rightNode):
    n0 = len(node.data.index)
    n1 = len(leftNode.data.index)
    n2 = len(rightNode.data.index)
    return node.entropy - n1 / n0 * leftNode.entropy - n2 / n0 * rightNode.entropy


def splitNode(node, usedParam, unusedParam, cur_depth):
    node.optimalSplitNode(unusedParam)
    usedParam.append(node.param)
    unusedParam.remove(node.param)
    node.left.depth = node.right.depth = cur_depth + 1
    return node.left, node.right, usedParam, unusedParam

def splitOneRound(tree, usedParam, unusedParam):
    max_depths = max([node.depth for node in tree])
    for node in tree:
        if node.property == 'leafNode' and node.label is None and node.depth < max_depths:
            left, right, used_attrs, unused_attrs = splitNode(node, usedParam=usedParam, unusedParam=unusedParam,
                                                              cur_depth=node.depth)
            tree.extend([left, right])
            return tree, used_attrs, unused_attrs
    for node in tree:
        if node.property == 'leafNode' and node.label is None:
            left, right, used_attrs, unused_attrs = splitNode(node, usedParam=usedParam, unusedParam=unusedParam,
                                                              cur_depth=node.depth)
            tree.extend([left, right])
            return tree, used_attrs, unused_attrs

def trainStatus(tree):
    for node in tree:
        if node.property == 'leafNode' and node.label is None:
            return False
    return True

def terminateTrain(tree):
    for idx, node in enumerate(tree):
        if node.property == 'leafNode' and node.label is None:
            node.makeTerminal()
            tree[idx] = node
    return tree

def buildTree(train_data):
    unused_attrs = list(train_data.columns.values)[:-1]
    used_attrs = list()

    tree = list()
    tree.append(Node(data=train_data, name='H', depth=0))
    left, right, used_attrs, unused_attrs = splitNode(tree[0], usedParam=used_attrs, unusedParam=unused_attrs,
                                                      cur_depth=0)
    tree.extend([left, right])
    while trainStatus(tree) is False:
        tree, used_attrs, unused_attrs = splitOneRound(tree, used_attrs, unused_attrs)
        if len(unused_attrs) == 0:
            tree = terminateTrain(tree)
    return tree

def searchLabel(node, newentry):
    attr = node.name.split('-')[-1].strip('0').strip('1')
    value = int(node.name.split('-')[-1][-1])
    data_value = newentry[attr]
    label = None
    if value == data_value:
        if node.label is not None:
            return node.label
        label = searchLabel(node.left, newentry)
        if label is not None:
            return label
        else:
            label = searchLabel(node.right, newentry)
            return label
    else:
        return label

def predictOneRow(tree, newentry):
    headNode = tree[0]
    label = searchLabel(headNode.left, newentry)
    if label is None:
        label = searchLabel(headNode.right, newentry)
    return label

def cal_accuracy(truth, prediction):
    correct = 0
    for i, j in zip(truth, prediction):
        if i == j:
            correct += 1
    return correct / len(truth)

def pruneOneNode(tree):
    leaf_idx = [idx for idx, node in enumerate(tree) if node.property == 'leafNode']
    prune_idx = int(np.random.choice(leaf_idx, 1))
    prune_node = tree[prune_idx]
    tree = [node for idx, node in enumerate(tree) if idx != leaf_idx]
    prune_name = prune_node.name
    parent_name = '-'.join(prune_name.split('-')[:-1])
    for idx, node in enumerate(tree):
        if node.name == parent_name:
            parent_idx = idx
    tree[parent_idx].makeTerminal()
    return tree


model = DecisionTree(train_data)
print('Pre-Pruned Decision Tree\n--------------------')
model.printTree()
model.preAccuracy(validation_data, test_data)
model.prune(factor=pruning_factor)
print('Post-Pruned Decision Tree\n--------------------')
model.printTree(prune=True)
model.postAccuracy(validation_data, test_data)