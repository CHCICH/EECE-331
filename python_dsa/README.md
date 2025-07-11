# Using the Python DSA Package


## Installation
Install the package via pip:
```bash
pip install gsl_331
```

This package exposes a few classes to manage common data structures. The available classes are:
- Queue
- Stack
- Graph
- Node
- LLNode
- LinkedList
- Priority_queue

Below are usage examples for each.

## Queue
Use the Queue class to enqueue and dequeue items.
```python
from gsl_331 import Queue

q = Queue()
q.enqueue("first")
q.enqueue("second")
print("Queue front:", q.top())  # prints "first"
print("Dequeued:", q.dequeue())
print("Queue is empty:", q.is_empty())
```

## Stack
The Stack class provides LIFO operations.
```python
from gsl_331 import Stack

s = Stack()
s.push("item1")
s.push("item2")
print("Stack top:", s.top())  # prints "item2"
print("Popped item:", s.pop())
print("Is stack empty:", s.is_empty())
```

## Graph and Node
Create a graph, add vertices via random generation, and run BFS/DFS. Both the BFS and DFS operations return custom response objects. The BFS operation returns a BFS_response containing visited nodes, visit order, and parent relationships. Similarly, the DFS provides a DFS_response with entry and exit times, pending nodes, and a stack with the order of the processed nodes.
```python
from gsl_331 import Graph, Node

# Create a directed graph with labels generated for n_node vertices.
g = Graph(n=5, directed=True)
g.generate_random_graph(n_node=5)

# Choose a starting node label (e.g., "a")
bfs_response = g.BFS("a")
print(bfs_response)  # prints visited nodes, order, and parent relationships given as a whole object of type bfs response

dfs_response = g.DFS("a")
print(dfs_response)  # prints visited nodes, entry/exit times, pending nodes, and processing stack given as a whole object of type bfs response
```

## LLNode and LinkedList
Manage a linked list with LLNode and LinkedList classes.
```python
from gsl_331 import LLNode, LinkedList

# Create the head LLNode and initialize the linked list.
head = LLNode(val=10, name="10")
ll = LinkedList(head)

# Insert values
ll.insertAtHead(5)
ll.insertAtTail(15)
print("Linked List:", ll)
print("Search for 15:", ll.search(15))
ll.delete(10)
print("After deleting 10:", ll)
```

## Priority_queue
The Priority_queue supports heap operations with fast lookup by node name. Its responses include returning the top element and providing means to extract a specific node.
```python
from gsl_331 import Priority_queue

# Initialize priority queue with a list of elements.
data = [4, 2, 6, 3]
pq = Priority_queue(data)

print("Priority Queue top:", pq.top())
pq.push(1, name="unique1")
print("Priority Queue after push:", pq)
extracted = pq.exctract_node("unique1")
print("Extracted node value:", extracted.val)
```
# NOTE 
that the priority queue contains a dictionary in pq.location_tracker that gives you the real time postions of objects from your pq and makes the extraction of any node O(logn) same as all other operations
