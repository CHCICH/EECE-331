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
class response:
    def __init__(self,type):
        self.type = type
class BFS_response(response):
    def __init__(self, visited=None, parents=None, ordered_visits=None):
        super().__init__("BFS_MAP")
        self.visited = visited if visited is not None else {}
        self.parents = parents if parents is not None else []
        self.ordered_visits = ordered_visits if ordered_visits is not None else []
    
    def __str__(self):
        result = f"BFS Response: BFS_MAP \n"
        result += f"Visited nodes: {self.visited}\n"
        result += f"Visit order: {self.ordered_visits}\n"
        result += f"Parent relationships: {self.parents}"
        return result

class DFS_response(response):
    def __init__(self,time=0, is_pending=None, visited=None, entry_time=None, exit_time=None, stack=None):
        super().__init__("DFS_MAP")
        self.is_pending = is_pending if is_pending is not None else {}
        self.visited = visited if visited is not None else []
        self.entry_time = entry_time if entry_time is not None else []
        self.exit_time = exit_time if exit_time is not None else []
        self.stack = stack if stack is not None else []
        self.time = time
    
    def __str__(self):
        result = f"DFS Response: DFS_MAP\n"
        result += f"Pending nodes: {self.is_pending}\n"
        result += f"Visited nodes: {self.visited}\n"
        result += f"Entry times: {self.entry_time}\n"
        result += f"Exit times: {self.exit_time}\n"
        result += f"Stack: {self.stack}\n"
        result += f"Time: {self.time}\n"

        return result
    
class Edges_type(response):
    def __init__(self,edge_type,edges, weight):
        super().__init__("EDGES_OBJECT")
        self.edge_type = edge_type if edge_type is not None else {}
        u,v = edges
        self.parent = u
        self.child = v
        self.weight = weight

    def __str__(self):
        result = ""
        result += f"edge w = { self.weight}, {self.parent} -> {self.child} is type: {self.edge_type} "
        return result


class Queue(queue.Queue):
    def enqueue(self, item):
        self.put(item)

    def dequeue(self):
        return self.get()

    def is_empty(self):
        return self.empty()

    def size(self):
        return self.qsize()
    
    def top(self):
        if not self.empty():
            item = self.get()
            self.put(item)
            return item
        return None

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
    
    def top(self):
        if not self.is_empty():
            return self.items[-1]
        return None
    def __str__(self):
        return f"{self.items}"


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
# the class Node has only a proprty of value which can be anything the type is to easily overload the Node 
# when we need to use it instead of overloading the whole class it seems redundant but better to take precautions

    

class Node:
    def __init__(self, val="a",name=0):
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
    def __init__(self,head):
        assert isinstance(head, LLNode), "head must be an LLNode"
        self.head = head
        self.tail = None

    def insertAtHead(self, elem):
        new_node = LLNode(val=elem, next=self.head, name=str(elem))
        self.head = new_node
        if self.tail is None:
            self.tail = new_node

    def insertAtTail(self, elem):
        new_node = LLNode(val=elem, next=None, name=str(elem))
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node

    def search(self, target):
        current = self.head
        while current:
            if current.val == target:
                return current
            current = current.next
        return None

    def delete(self, target):
        current = self.head
        prev = None
        while current:
            if current.val == target:
                if prev:
                    prev.next = current.next
                else:
                    self.head = current.next
                if current == self.tail:
                    self.tail = prev
                return True
            prev = current
            current = current.next
        return False

    def __str__(self):
        curr = self.head
        final = ""
        while curr:
            final += str(curr.val) + "->"
            curr = curr.next
        return final

class edges:
    def __init__(self,node,weight=0):
        self.node = node
        self.weight = weight
    

class Graph:
    def __init__(self,n=10,min_weight=0, max_weight=10,directed=True):
        self.edges = {}
        self.directed = directed
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
    def BFS(self,starting_node,visited_prev={}):
        q = Queue()
        visited = visited_prev
        ordered_visits = []
        parents_array = [] 
        q.enqueue(starting_node)
        visited[starting_node] = True
        while not q.empty():
            current_node = q.top()
            q.dequeue()
            ordered_visits.append(current_node)
            for u in self.vertcies[current_node]:
                if u[0] not in visited:
                    q.enqueue(u[0])
                    visited[u[0]] = True
                    parents_array.append((current_node,u[0]))

        return BFS_response(visited,parents_array,ordered_visits)


    def DFS(self,starting_node="",visted_prev={}):
        is_pending = {}
        visited = visted_prev
        entry_time = {}
        exit_time = {}
        time = 0
        stack = Stack()
        def DFSearch(self, u, time, is_pending, visited, entry_time, exit_time, stack):
            time += 1
            entry_time[u] = time
            is_pending[u] = True
            for (v,w) in self.vertcies[u]:
                if v not in is_pending:
                    time, is_pending, visited, entry_time, exit_time, stack = DFSearch(G, v, time, is_pending, visited, entry_time, exit_time, stack)
            time += 1
            exit_time[u] = time
            visited[u] = True
            stack.push(u)
            return [time, is_pending, visited, entry_time, exit_time, stack]
        if(starting_node == ""):
            for u in self.vertcies.keys():
                if u not in visited:
                    time, is_pending, visited, entry_time, exit_time, stack = DFSearch(self, u, time, is_pending, visited, entry_time, exit_time, stack)
        else:
            time, is_pending, visited, entry_time, exit_time, stack = DFSearch(G, starting_node, time, is_pending, visited, entry_time, exit_time, stack)

        return DFS_response(time, is_pending, visited, entry_time, exit_time, stack)
