'''Author: CHICH'''
import queue
import heapq
import random
# this function allows us to write clearner code and to manage min and max structures

def less_important(item1, item2,is_max ):
    if is_max:
        return item1 > item2
    return item1 < item2


# to avoid repetion we are not going to reimplement the class queue but just rename their components aka the names of the function that do 
# not align with the classical names

class Queue(queue.Queue):
    def enqueue(self, item):
        self.put(item)

    def dequeue(self):
        return self.get()

    def is_empty(self):
        return self.empty()

    def size(self):
        return self.qsize()

# implementation of stacks is pretty straight forward 
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        return None

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)


# the heapq class is going to be the same because the heapq allows us to only modify the 
# but we are going to totally dump this premade architecture because of the need to implement a quick search query of the position of a node and delete it
# which is going to require us to keep track of all nodes and their position in a dictionary this will enable us to garentee better alogrithms such as 
# a faster Dijkstra Algorithm

class Priority_queue:
    def __init__(self,data = [],is_fix=False):
        self.data =[]
        if not is_fix:
            self.data = [Node(data[i], i) for i in range(len(data))]
            heapq.heapify(self.data)  # Initialize and heapify self.data
        else:
            self.data = data
            heapq.heapify(self.data)
        self.location_tracker = {}
        for i in range(len(self.data)):
            elem = self.data[i]
            self.location_tracker[elem.name] = i
        self.max_val = len(data) -1 # this variable never decreases and refers to the id to the final element to avoid collisions 
        self.is_max = False

    def bubble_down(self, node):
        if len(self.data) == 0:
            return False
        index = self.location_tracker[node.name]
        def helper_bubble_down(index):
            child1 = 2*index + 1
            child2 = 2* index + 2
            if (child1 > len(self.data) - 1) or (child2 > len(self.data) - 1):
                return True
            
            if (less_important(self.data[child1],self.data[index],self.is_max)) or less_important(self.data[child2],self.data[index],self.is_max):
                if (child2 > len(self.data) - 1):
                    self.location_tracker[node.name] = child1
                    self.location_tracker[self.data[child1].name] = index
                    (self.data[index],self.data[child1]) = (self.data[child1],self.data[index])
                    return helper_bubble_down(child1)
                else:
                    current_child = None
                    if(less_important(self.data[child1],self.data[child2],self.is_max)):
                        current_child = child1
                    else:
                        current_child = child2
                    self.location_tracker[node.name] = current_child
                    self.location_tracker[self.data[current_child].name] = index
                    (self.data[index],self.data[current_child]) = (self.data[current_child],self.data[index])
                    return helper_bubble_down(current_child)
            return True
        
        return helper_bubble_down(index)
    
    def bubble_up(self,node):
        if len(self.data) == 0:
            return False
        index = self.location_tracker[node.name]
        def helper_bubble_down(index):
            if index == 0:
                return True
            parent = int((index-1)/2)
            if(less_important(self.data[index],self.data[parent],self.is_max)):
                self.location_tracker[node.name] = parent
                self.location_tracker[self.data[parent].name] = index
                (self.data[parent],self.data[index]) = (self.data[index],self.data[parent])
                return helper_bubble_down(parent)
            
            return True
        
        return helper_bubble_down(index)
    def push(self,elem,name):
        self.max_val +=1
        new_node = Node(elem,name)
        self.data.append(new_node)
        self.location_tracker[name] = len(self.data) -1
        self.bubble_up(new_node)
        return True
    def top(self):
        if len(self.data) == 0:
            return False
        return self.data[0].val
    def extract_top(self):
        if len(self.data) == 0:
            return False
        (self.data[0], self.data[len(self.data) -1]) = (self.data[len(self.data) -1], self.data[0])
        self.location_tracker[self.data[0].name] = 0
        self.location_tracker[self.data[len(self.data)-1].name] = len(self.data)-1
        self.data.pop()
        self.bubble_down(self.data[0])
    def exctract_node(self,name):
        if len(self.data) == 0:
            return False
        index = self.location_tracker[name]
        (self.data[index], self.data[len(self.data) -1]) = (self.data[len(self.data) -1], self.data[index])
        self.location_tracker[self.data[index].name] = 0
        self.location_tracker[self.data[len(self.data)-1].name] = len(self.data)-1
        finalNode = self.data[len(self.data)-1]
        self.data.pop()
        self.bubble_down(self.data[index])
        return finalNode
    def __str__(self):
        final_arr = []
        for elem in self.data:
            final_arr.append(elem.val)
        return str(final_arr)
    
# a graph is either A directed or undirected
# and also the style of storing the elements and the nodes in a graph differ from one to another 
# the class Node has only a proprty of value which can be anything the reason is to easily overload the Node 
# when we need to use it instead of overloading the whole class it seems redundant but better to take precautions

    

class Node:
    def __init__(self, name="a",val=0):
        self.val = val
        self.name = name

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.val == other.val and self.name == other.name
        return False

    def __hash__(self):
        return hash((self.val, self.name))

    def __ge__(self, other):
        if isinstance(other, Node):
            return self.val >= other.val
        return False

    def __le__(self, other):
        if isinstance(other, Node):
            return self.val <= other.val
        return False

    def __lt__(self, other):
        if isinstance(other, Node):
            return self.val < other.val
        return False

    def __gt__(self, other):
        if isinstance(other, Node):
            return self.val > other.val
        return False
    def __str__(self):
        return str(self.val)
     
class LLNode(Node):
    def __init__(self, val=0, next=None, name="a"):
        super().__init__(val, name)
        self.next = next
# graph class

class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    def insertAtHead(self,elem):
        if self.head == None:
            self.head = LLNode(elem)
            self.tail = LLNode(elem)
        else:       
            new_node = LLNode(elem)
            new_node.next = self.head
            self.head = new_node
    def insertAfterNode(self,name):
        current = self.head
        while(current != None):
            if current.name == name:

                current = current.next

class edges:
    def __init__(self,node,weight=0):
        self.node = node
        self.weight = weight
    

class Graph:
    def __init__(self,n=10,min_weight=0, max_weight=10):
        self.edges = {}
        self.labels = [str(i) for i in range(n)]
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.vertcies = {}

    def generate_random_graph(self, n_node=0):
        def generate_label(index):
            label = ""
            while index >= 0:
                label = chr(index % 26 + ord('a')) + label
                index = index // 26 - 1
            return label

        self.labels = [generate_label(i) for i in range(n_node)]
        for vertex in self.labels:
            self.vertcies[vertex] = []
        for i in range(n_node):
            for j in range(random.randint(1, n_node - 1)):
                from_node = Node(self.labels[i])
                to_node = Node(random.choice([label for label in self.labels if label != from_node.name]))
                weight = random.randint(self.min_weight, self.max_weight)
                triplet = (from_node, to_node, weight)
                if triplet not in self.edges.values():
                    self.edges[len(self.edges)] = triplet
                    self.vertcies[from_node.name].append((to_node.name,weight))

    # setting the labels in other words the tags for each node also known as their names 

    def set_labels(self,lables):
        self.labels = lables


    # this needs to be modified and correctly set to match with the other one  
    def set_values(self,val_matrix):
        for i in range(len(val_matrix)):
            self.edges[Node(self.labels[i])] = Node(val_matrix[i])
        return True
    
    # this would return the parent and vists array
    def BFS():
        return # for now retruns nothing

    

G = Graph()
G.generate_random_graph(3)
print(G.vertcies,G.labels)