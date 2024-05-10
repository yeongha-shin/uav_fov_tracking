"""
@file: lazy_theta_star.py
@breif: Lazy Theta* motion planning
@author: Yang Haodong, Wu Maojia
@update: 2024.2.11
"""
import heapq

from .theta_star import ThetaStar
from tool.utils import Env, Node

class LazyThetaStar(ThetaStar):
    """
    Class for Lazy Theta* motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Env): environment
        heuristic_type (str): heuristic function type

    Examples:
        >>> from tool.utils import Grid
        >>> from graph_search import LazyThetaStar
        >>> start = (5, 5)
        >>> goal = (45, 25)
        >>> env = Grid(51, 31)
        >>> planner = LazyThetaStar(start, goal, env)
        >>> planner.run()

    References:
        [1] Lazy Theta*: Any-Angle Path Planning and Path Length Analysis in 3D
    """
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type)

    def __str__(self) -> str:
        return "Lazy Theta*"

    def plan(self):
        """
        Lazy Theta* motion plan function.

        Returns:
            cost (float): path cost
            path (list): planning path
            expand (list): all nodes that planner has searched
        """
        # OPEN set with priority and CLOSED set
        OPEN = []
        heapq.heappush(OPEN, self.start)
        CLOSED = []

        while OPEN:
            node = heapq.heappop(OPEN)

            # set vertex: path 1
            try:
                node_p = CLOSED[CLOSED.index(Node(node.parent))]
                if not self.lineOfSight(node_p, node):
                    node.g = float("inf")
                    for node_n in self.getNeighbor(node):
                        if node_n in CLOSED:
                            node_n = CLOSED[CLOSED.index(node_n)]
                            if node.g > node_n.g + self.dist(node_n, node):
                                node.g = node_n.g + self.dist(node_n, node)
                                node.parent = node_n.current
            except:
                pass

            # exists in CLOSED set
            if node in CLOSED:
                continue

            # goal found
            if node == self.goal:
                CLOSED.append(node)
                cost, path = self.extractPath(CLOSED)
                return cost, path, CLOSED

            for node_n in self.getNeighbor(node):                
                # exists in CLOSED set
                if node_n in CLOSED:
                    continue
                
                # path1
                node_n.parent = node.current
                node_n.h = self.h(node_n, self.goal)

                try:
                    p_index = CLOSED.index(Node(node.parent))
                    node_p = CLOSED[p_index]
                except:
                    node_p = None

                if node_p:
                    # path2
                    self.updateVertex(node_p, node_n)

                # goal found
                if node_n == self.goal:
                    heapq.heappush(OPEN, node_n)
                    break
                
                # update OPEN set
                heapq.heappush(OPEN, node_n)
            
            CLOSED.append(node)
        return [], [], []
    
    def updateVertex(self, node_p: Node, node_c: Node) -> None:
        """
        Update extend node information with current node's parent node.

        Parameters:
            node_p (Node): parent node
            node_c (Node): current node
        """
        # path 2
        if node_p.g + self.dist(node_c, node_p) <= node_c.g:
            node_c.g = node_p.g + self.dist(node_c, node_p)
            node_c.parent = node_p.current  
