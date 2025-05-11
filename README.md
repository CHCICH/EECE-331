
# Custom Data Structures: Queue, Priority Queue, Node, Graph, and Red-Black Tree

This module provides implementations for a custom `Queue` class, a `priority_queue` class, a `Node` class, a `Graph` class, and a `Red-Black Tree`. Below are usage examples and details for each class.

---

## Queue Class

The `Queue` class is a subclass of `queue.Queue` with renamed methods for better alignment with classical naming conventions.

### Methods:
- **`enqueue(item)`**: Adds an item to the queue.
- **`dequeue()`**: Removes and returns an item from the queue.
- **`is_empty()`**: Checks if the queue is empty.
- **`size()`**: Returns the number of items in the queue.

### Example Usage:
```python
from your_module import Queue

# Create a queue
q = Queue()

# Add items to the queue
q.enqueue(10)
q.enqueue(20)

# Check if the queue is empty
print(q.is_empty())  # Output: False

# Get the size of the queue
print(q.size())  # Output: 2

# Remove an item from the queue
print(q.dequeue())  # Output: 10
```

---

## Priority Queue Class

The `priority_queue` class is a custom implementation of a priority queue that supports efficient node management and quick search queries.

### Constructor:
- **`priority_queue(data=[], is_fix=True)`**: Initializes the priority queue. If `is_fix` is `False`, the queue is initialized with `Node` objects.

### Methods:
- **`push(elem, name)`**: Adds a new node with the given value and name.
- **`top()`**: Returns the value of the top node without removing it.
- **`extract_top()`**: Removes and returns the top node.
- **`exctract_node(name)`**: Removes and returns a node by its name.

### Example Usage:
```python
from your_module import priority_queue

# Create a priority queue
pq = priority_queue()

# Add elements
pq.push(10, "A")
pq.push(20, "B")

# Get the top element
print(pq.top())  # Output: 10

# Extract the top element
pq.extract_top()

# Extract a specific node
node = pq.exctract_node("B")
print(node.val)  # Output: 20
```

---

## Node Class

The `Node` class represents a node with a value and supports comparison operators.

### Constructor:
- **`Node(val=0, name="a")`**: Initializes a node with a value and a name.

### Comparison Operators:
- **`__eq__`**, **`__ge__`**, **`__le__`**, **`__lt__`**, **`__gt__`**: Allow comparison of nodes based on their values.

### Example Usage:
```python
from your_module import Node

# Create nodes
node1 = Node(val=5, name="A")
node2 = Node(val=10, name="B")

# Compare nodes
print(node1 < node2)  # Output: True
print(node1 == node2)  # Output: False
```

---

## Graph Class

The `Graph` class represents a graph with labeled nodes and edges.

### Constructor:
- **`Graph(n=10)`**: Initializes a graph with `n` nodes (default is 10). Nodes are labeled as strings of integers.

### Methods:
- **`set_labels(labels)`**: Sets custom labels for the nodes.
- **`set_values(val_matrix)`**: Sets the values of nodes based on a given matrix, mapping labels to their corresponding values.

### Example Usage:
```python
from your_module import Graph

# Create a graph with 5 nodes
g = Graph(n=5)

# Set custom labels
g.set_labels(["A", "B", "C", "D", "E"])

# Set values for the nodes
value_matrix = [1, 2, 3, 4, 5]
g.set_values(value_matrix)

# Access graph properties
print(g.labels)  # Output: ['A', 'B', 'C', 'D', 'E']
print(g.edges)   # Output: {Node('A'): Node(1), Node('B'): Node(2), ...}
```

---

## Red-Black Tree

This implementation of the Red-Black Tree comes from Geeks for Geeks. Personally, we didn't find the need to correct some implementation and no customization was made.

### Classes:
- **`RBNode`**: Represents a node in the Red-Black Tree.
- **`RedBlackTree`**: Implements the Red-Black Tree with insertion, deletion, and search functionalities.

### Example Usage:
```python
from your_module import RedBlackTree

# Create a Red-Black Tree
rbt = RedBlackTree()

# Insert elements
rbt.insert(10)
rbt.insert(20)
rbt.insert(15)

# Search for an element
node = rbt.search(15)
print(node.value)  # Output: 15

# Delete an element
rbt.delete(10)
```

### Key Features:
- **Balanced Binary Search Tree**: Ensures logarithmic height for efficient operations.
- **Automatic Rebalancing**: Handles rotations and color changes during insertions and deletions.

---

## Notes:
- The `Queue` class simplifies method names for better readability.
- The `priority_queue` class is designed for efficient node management and supports both min-heap and max-heap configurations.
- The `Node` class is designed to be extensible for advanced use cases.
- The `Graph` class supports flexible node and edge management for both directed and undirected graphs.
- The `Red-Black Tree` implementation is based on a standard algorithm and is suitable for maintaining balanced binary search trees.