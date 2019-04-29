from random import *
from math import *

'''
Tree Node 
 - each node has an associated weight 

Author: Ly Sung
Date: March 23rd 2019
'''

class Node():
    value = -1
    parent = None
    index = 999999

    def __init__(self, parent = None, left = None, right = None, bla=0):
        self.left = left
        self.right = right
        self.weight = round(random(), 2)
        self.parent = parent

    '''
    @TODO: update weights 
    '''
    def update_weight(self, sum_error, input, c = 0.1):
        new_weight = (sum_error * input * c + self.weight)
        self.weight = new_weight

    def set_left_node(self, node):
        self.left = node

    def set_right_node(self, node):
        self.right = node

    def set_value(self, value):
        self.value = value

    def set_index(self, index):
        self.index = index

    def has_left(self):
        if self.left:
            return True
        return False

    def has_right(self):
        if self.right:
            return True
        return False

    def get_index(self):
        return self.index

    def get_value(self):
        return self.value