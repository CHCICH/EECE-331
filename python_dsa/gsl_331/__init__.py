"""
GSL-331 Data Structures Library

A comprehensive Python library providing custom implementations of essential data structures
and algorithms including queues, stacks, priority queues, graphs, linked lists, and more.

Author: CHICH
"""

from .core import (
    Queue,
    Stack,
    Priority_queue,
    Node,
    LLNode,
    LinkedList,
    Graph,
    BFS_response,
    DFS_response,
    Edges_type,
    less_important
)

__version__ = "1.0.0"
__author__ = "CHICH"

__all__ = [
    'Queue',
    'Stack', 
    'Priority_queue',
    'Node',
    'LLNode',
    'LinkedList',
    'Graph',
    'BFS_response',
    'DFS_response',
    'Edges_type',
    'less_important'
]
