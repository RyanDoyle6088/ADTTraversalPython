"""
Name:
CSE 331 FS20 (Onsay)
"""

import heapq
import itertools
import math
import queue
import random
import time
from typing import TypeVar, Callable, Tuple, List, Set

import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

T = TypeVar('T')
Matrix = TypeVar('Matrix')  # Adjacency Matrix
Vertex = TypeVar('Vertex')  # Vertex Class Instance
Graph = TypeVar('Graph')    # Graph Class Instance


class Vertex:
    """ Class representing a Vertex object within a Graph """

    __slots__ = ['id', 'adj', 'visited', 'x', 'y']

    def __init__(self, idx: str, x: float = 0, y: float = 0) -> None:
        """

        Initializes a Vertex
        :param idx: A unique string identifier used for hashing the vertex
        :param x: The x coordinate of this vertex (used in a_star)
        :param y: The y coordinate of this vertex (used in a_star)
        """
        self.id = idx
        self.adj = {}             # dictionary {id : weight} of outgoing edges
        self.visited = False      # boolean flag used in search algorithms
        self.x, self.y = x, y     # coordinates for use in metric computations

    def __eq__(self, other: Vertex) -> bool:
        """
        Equality operator for Graph Vertex class
        :param other: vertex to compare
        """
        if self.id != other.id:
            return False
        elif self.visited != other.visited:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex visited flags not equal: self.visited={self.visited},"
                  f" other.visited={other.visited}")
            return False
        elif self.x != other.x:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex x coords not equal: self.x={self.x}, other.x={other.x}")
            return False
        elif self.y != other.y:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex y coords not equal: self.y={self.y}, other.y={other.y}")
            return False
        elif set(self.adj.items()) != set(other.adj.items()):
            diff = set(self.adj.items()).symmetric_difference(set(other.adj.items()))
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex adj dictionaries not equal:"
                  f" symmetric diff of adjacency (k,v) pairs = {str(diff)}")
            return False
        return True

    def __repr__(self) -> str:
        """
        Represents Vertex object as string.
        :return: string representing Vertex object
        """
        lst = [f"<id: '{k}', weight: {v}>" for k, v in self.adj.items()]

        return f"<id: '{self.id}'" + ", Adjacencies: " + "".join(lst) + ">"

    def __str__(self) -> str:
        """
        Represents Vertex object as string.
        :return: string representing Vertex object
        """
        return repr(self)

    def __hash__(self) -> int:
        """
        Hashes Vertex into a set; used in unit tests
        :return: hash value of Vertex
        """
        return hash(self.id)

    def degree(self) -> int:
        """
        Returns the number of outgoing edges from this vertex
        :return: int of degree
        """
        return len(self.adj)

    def get_edges(self) -> Set[Tuple[str, float]]:
        """
        Returns a set of tuples representing outgoing edges from this vertex.
        :return: set of tuples
        """
        edges = set()
        for i in self.adj:
            edges.add((i, self.adj[i]))
        return edges

    def euclidean_distance(self, other: Vertex) -> float:
        """
        Returns the euclidean distance
        between this vertex and vertex other.
        :param other: vertex
        :return: flt
        """
        x1 = self.x
        y1 = self.y
        x2 = other.x
        y2 = other.y
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def taxicab_distance(self, other: Vertex) -> float:
        """
        Returns the taxicab distance
        between this vertex and vertex other.
        :param other: vertex
        :return: flt
        """
        x1 = self.x
        y1 = self.y
        x2 = other.x
        y2 = other.y
        return abs(x1 - x2) + abs(y1 - y2)


class Graph:
    """ Class implementing the Graph ADT using an Adjacency Map structure """

    __slots__ = ['size', 'vertices', 'plot_show', 'plot_delay']

    def __init__(self, plt_show: bool = False, matrix: Matrix = None, csv: str = "") -> None:
        """
        Instantiates a Graph class instance
        :param: plt_show : if true, render plot when plot() is called; else, ignore calls to plot()
        :param: matrix : optional matrix parameter used for fast construction
        :param: csv : optional filepath to a csv containing a matrix
        """
        matrix = matrix if matrix else np.loadtxt(csv, delimiter=',', dtype=str).tolist() if csv else None
        self.size = 0
        self.vertices = {}

        self.plot_show = plt_show
        self.plot_delay = 0.2

        if matrix is not None:
            for i in range(1, len(matrix)):
                for j in range(1, len(matrix)):
                    if matrix[i][j] == "None" or matrix[i][j] == "":
                        matrix[i][j] = None
                    else:
                        matrix[i][j] = float(matrix[i][j])
            self.matrix2graph(matrix)

    def __eq__(self, other: Graph) -> bool:
        """
        Overloads equality operator for Graph class
        :param other: graph to compare
        """
        if self.size != other.size or len(self.vertices) != len(other.vertices):
            print(f"Graph size not equal: self.size={self.size}, other.size={other.size}")
            return False
        else:
            for vertex_id, vertex in self.vertices.items():
                other_vertex = other.get_vertex(vertex_id)
                if other_vertex is None:
                    print(f"Vertices not equal: '{vertex_id}' not in other graph")
                    return False

                adj_set = set(vertex.adj.items())
                other_adj_set = set(other_vertex.adj.items())

                if not adj_set == other_adj_set:
                    print(f"Vertices not equal: adjacencies of '{vertex_id}' not equal")
                    print(f"Adjacency symmetric difference = "
                          f"{str(adj_set.symmetric_difference(other_adj_set))}")
                    return False
        return True

    def __repr__(self) -> str:
        """
        Represents Graph object as string.
        :return: String representation of graph for debugging
        """
        return "Size: " + str(self.size) + ", Vertices: " + str(list(self.vertices.items()))

    def __str__(self) -> str:
        """
        Represents Graph object as string.
        :return: String representation of graph for debugging
        """
        return repr(self)

    def plot(self) -> None:
        """
        Creates a plot a visual representation of the graph using matplotlib
        :return: None
        """
        if self.plot_show:

            # if no x, y coords are specified, place vertices on the unit circle
            for i, vertex in enumerate(self.get_vertices()):
                if vertex.x == 0 and vertex.y == 0:
                    vertex.x = math.cos(i * 2 * math.pi / self.size)
                    vertex.y = math.sin(i * 2 * math.pi / self.size)

            # show edges
            num_edges = len(self.get_edges())
            max_weight = max([edge[2] for edge in self.get_edges()]) if num_edges > 0 else 0
            colormap = cm.get_cmap('cool')
            for i, edge in enumerate(self.get_edges()):
                origin = self.get_vertex(edge[0])
                destination = self.get_vertex(edge[1])
                weight = edge[2]

                # plot edge
                arrow = patches.FancyArrowPatch((origin.x, origin.y),
                                                (destination.x, destination.y),
                                                connectionstyle="arc3,rad=.2",
                                                color=colormap(weight / max_weight),
                                                zorder=0,
                                                **dict(arrowstyle="Simple,tail_width=0.5,"
                                                                  "head_width=8,head_length=8"))
                plt.gca().add_patch(arrow)

                # label edge
                plt.text(x=(origin.x + destination.x) / 2 - (origin.x - destination.x) / 10,
                         y=(origin.y + destination.y) / 2 - (origin.y - destination.y) / 10,
                         s=weight, color=colormap(weight / max_weight))

            # show vertices
            x = np.array([vertex.x for vertex in self.get_vertices()])
            y = np.array([vertex.y for vertex in self.get_vertices()])
            labels = np.array([vertex.id for vertex in self.get_vertices()])
            colors = np.array(
                ['yellow' if vertex.visited else 'black' for vertex in self.get_vertices()])
            plt.scatter(x, y, s=40, c=colors, zorder=1)

            # plot labels
            for j, _ in enumerate(x):
                plt.text(x[j] - 0.03*max(x), y[j] - 0.03*max(y), labels[j])

            # show plot
            plt.show()
            # delay execution to enable animation
            time.sleep(self.plot_delay)


    def reset_vertices(self) -> None:
        """
        Resets visited flags of all vertices within the graph.
        :return: None
        """
        for i in self.vertices:
            obj = self.vertices[i]
            obj.visited = False

    def get_vertex(self, vertex_id: str) -> Vertex:
        """
        Returns the Vertex object with id vertex_id if it exists in the graph,
        or None.
        :param vertex_id: vertex
        :return: Vertex object from graph
        """
        return self.vertices.get(vertex_id)

    def get_vertices(self) -> Set[Vertex]:
        """
        Returns a set of all Vertex objects held in the graph,
        or an empty set.
        :return: Set of vertices or empty
        """
        all_vertices = set()
        for i in self.vertices:
            all_vertices.add(self.vertices[i])
        return all_vertices

    def get_edge(self, start_id: str, dest_id: str) -> Tuple[str, str, float]:
        """
        Returns the edge connecting the vertex with id start_id to the
        vertex with id dest_id in a tuple of the form (start_id, dest_id, weight).
        :param start_id: vertex
        :param dest_id: vertex
        :return: tuple
        """
        wt = self.vertices.get(start_id)
        if wt is not None:
            wt = self.vertices.get(start_id).adj.get(dest_id)
            if wt is not None:
                return start_id, dest_id, wt
            else:
                return None
        else:
            return None

    def get_edges(self) -> Set[Tuple[str, str, float]]:
        """
        Returns a set of tuples representing all edges within the graph.
        :return: Set of tuples
        """
        all_edges = set()
        for start in self.vertices:
            for dest in self.vertices[start].adj:
                start_id = self.vertices[start].id
                dest_id = self.vertices[dest].id
                edge = self.get_edge(start_id,dest_id)
                all_edges.add(edge)
        return all_edges

    def add_to_graph(self, start_id: str, dest_id: str = None, weight: float = 0) -> None:
        """
        Adds a vertex / vertices / edge to the graph .
        :param start_id: vertex
        :param dest_id: vertex
        :param weight: Int weight of edge
        :return: None
        """
        if self.vertices.get(start_id) is None:
            self.vertices[start_id] = Vertex(start_id)
        if self.vertices.get(dest_id) is None and dest_id is not None:
            self.vertices[dest_id] = Vertex(dest_id)
        if dest_id is not None:
            self.vertices[start_id].adj[dest_id] = weight
        self.size = len(self.vertices)

    def matrix2graph(self, matrix: Matrix) -> None:
        """
        Constructs a graph from a given adjacency matrix representation.
        :param matrix:  a square 2D list
        :return: None
        """
        v = len(matrix)
        for i in range(1, v):
            self.vertices[matrix[0][i]] = Vertex(matrix[0][i])
        for i in range(1, v):
            for j in range(1, v):
                if matrix[i][j] is not None:
                    self.vertices[matrix[0][i]].x = i
                    self.vertices[matrix[0][j]].y = j
                    self.vertices[matrix[0][i]].adj[matrix[0][j]] = matrix[i][j]
        self.size = len(self.vertices)

    def graph2matrix(self) -> Matrix:
        """
        Constructs and returns an adjacency matrix from a graph.
        :return: adjacency matrix
        """
        v = len(self.vertices)
        if v == 0:
            return None
        matrix = [[None for i in range(v+1)] for j in range(v+1)]
        j = 1
        for i in self.vertices:
            if j < v+1:
                matrix[0][j] = i
                matrix[j][0] = i
                j += 1
        for start in self.vertices:
            for dest in self.vertices[start].adj:
                i = self.vertices[start].x
                j = self.vertices[dest].y
                wt = self.vertices[start].adj[dest]
                matrix[i][j] = wt
        return matrix

    def bfs(self, start_id: str, target_id: str) -> Tuple[List[str], float]:
        """
        breadth-first search algorithm
        :param start_id: vertex
        :param target_id: vertex
        :return: tuple of the form ([path], distance)
        """
        if self.vertices.get(start_id) is None or self.vertices.get(target_id) is None:
            return [], 0
        if self.vertices.get(start_id).adj.get(target_id) is not None:
            return [start_id, target_id], self.vertices.get(start_id).adj[target_id]
        q = queue.Queue()
        q.put(self.vertices[start_id])
        self.vertices[start_id].visited = True
        parent = [-1]*self.size
        dist = [0]*self.size
        e_dist = {}
        maps = {}
        j = 0
        e_dist[start_id] = 0
        for i in self.vertices:
            maps[i] = j
            j += 1
        path = []
        while q.empty() is not True:
            curr = q.get()
            if curr.id == target_id:
                temp = target_id
                while temp != -1:
                    path.append(temp)
                    temp = parent[maps[temp]]
                path.reverse()
                return path, dist[maps[target_id]]
            for i in self.vertices[curr.id].adj:
                if not self.vertices[i].visited:
                    q.put(self.vertices[i])
                    dist[maps[i]] = dist[maps[curr.id]] + self.vertices[curr.id].adj[i]
                    parent[maps[i]] = curr.id
                    self.vertices[i].visited = True
        return [], 0

    def dfs(self, start_id: str, target_id: str) -> Tuple[List[str], float]:
        """
        Depth first search algorithm
        :param start_id: vertex
        :param target_id: vertex
        :return: tuple of the form ([path], distance)
        """

        def dfs_inner(current_id: str, target_id: str, path: List[str] = [])\
                -> Tuple[List[str], float]:
            """
            Performs the recursive work of depth-first search by searching for a
            path from vertex with id current_id to vertex with id target_id.
            :param current_id: vertex
            :param target_id: vertex
            :param path: List of vertex IDs
            :return: Boolean
            """
            self.vertices[current_id].visited = True  # mark current node as discovered
            path.append(current_id)
            if current_id == target_id:  # if destination vertex is found
                return True
            for i in self.vertices[current_id].adj:
                if not self.vertices[i].visited:
                    if dfs_inner(i, target_id, path):
                        return True
            path.pop()
            return False
        path = []
        dist = 0
        if self.vertices.get(start_id) is None or self.vertices.get(target_id) is None:
            return [], 0
        if dfs_inner(start_id, target_id, path):
            for i in range(1, len(path)):
                dist += self.vertices[path[i - 1]].adj[path[i]]
            return path, dist
        else:
            return [], 0

    def a_star(self, start_id: str, target_id: str, metric: Callable[[Vertex, Vertex], float])\
            -> Tuple[List[str], float]:
        """
        Perform a A* search beginning at vertex with id start_id and
        terminating at vertex with id end_id
        :param start_id: vertex
        :param target_id: vertex
        :param metric: the remaining distance at each vertex
        :return: tuple of the form ([path], distance)
        """
        if self.vertices.get(start_id) is None or self.vertices.get(target_id) is None:
            return [], 0

        def reconstruct_path(came_from, start, end):
            """
            Helper function for search function
            """
            reverse_path = [end]
            while end != start:
                end = came_from[end]
                reverse_path.append(end)
            return list(reversed(reverse_path))
        frontier = []
        f_score = {self.vertices[i]: float("inf") for i in self.vertices}
        g_score = {self.vertices[i]: float("inf") for i in self.vertices}
        g_score[self.vertices[start_id]] = 0
        f_score[self.vertices[start_id]] = metric(self.vertices[start_id], self.vertices[target_id])
        count = 0
        heapq.heappush(frontier, (0, count, self.vertices[start_id]))
        came_from = dict()
        visited = set()
        while len(frontier) != 0:
            current = heapq.heappop(frontier)
            if current[2] in visited:
                continue
            if current[2].id == target_id:
                path = reconstruct_path(came_from, start_id, target_id)
                costs = 0
                for i in range(1, len(path)):
                    costs += self.vertices[path[i - 1]].adj[path[i]]
                return path, costs
            visited.add(current[2])
            for sec in self.vertices[current[2].id].adj:
                new_dist = g_score[current[2]] + self.vertices[current[2].id].adj[sec]
                if new_dist < g_score[self.vertices[sec]]:
                    came_from[sec] = current[2].id
                    g_score[self.vertices[sec]] = new_dist
                    f_score[self.vertices[sec]] = new_dist + metric(self.vertices[sec], self.vertices[target_id])
                    if self.vertices[sec] not in visited:
                        priority = f_score[self.vertices[sec]]
                        count += 1
                        heapq.heappush(frontier, (priority, count, self.vertices[sec]))
        return [], 0

    def make_equivalence_relation(self) -> int:
        """
        Determine if a given graph describes an equivalence relation.
        If not, add to the graph the minimum number of edges to make
        it an equivalence relation.
        :return: The number of edges added, or 0 if already eq.
        """
        matrix = self.graph2matrix()
        if matrix is None:
            return 0
        count = 0
        for i in range(1, len(matrix)):
            for j in range(1, len(matrix)):
                if i == j and matrix[i][j] != 1:
                    count += 1
                    matrix[i][j] = 1
                else:
                    if matrix[i][j] == 1:
                        if matrix[j][i] != 1:
                            count += 1
                            matrix[j][i] = 1
        for i in range(1, len(matrix)):
            for j in range(1, len(matrix)):
                if matrix[i][j] == 1:
                    for k in range(1, len(matrix)):
                        if matrix[j][k] == 1 and matrix[i][k] != 1:
                            count += 1
                            matrix[i][k] = 1
        self.matrix2graph(matrix)
        return count


class AStarPriorityQueue:
    """
    Priority Queue built upon heapq module with support for priority key updates
    Created by Andrew McDonald
    Inspired by https://docs.python.org/3/library/heapq.html
    """

    __slots__ = ['data', 'locator', 'counter']

    def __init__(self) -> None:
        """
        Construct an AStarPriorityQueue object
        """
        self.data = []                        # underlying data list of priority queue
        self.locator = {}                     # dictionary to locate vertices within priority queue
        self.counter = itertools.count()      # used to break ties in prioritization

    def __repr__(self) -> str:
        """
        Represent AStarPriorityQueue as a string
        :return: string representation of AStarPriorityQueue object
        """
        lst = [f"[{priority}, {vertex}], " if vertex is not None else "" for
               priority, count, vertex in self.data]
        return "".join(lst)[:-1]

    def __str__(self) -> str:
        """
        Represent AStarPriorityQueue as a string
        :return: string representation of AStarPriorityQueue object
        """
        return repr(self)

    def empty(self) -> bool:
        """
        Determine whether priority queue is empty
        :return: True if queue is empty, else false
        """
        return len(self.data) == 0

    def push(self, priority: float, vertex: Vertex) -> None:
        """
        Push a vertex onto the priority queue with a given priority
        :param priority: priority key upon which to order vertex
        :param vertex: Vertex object to be stored in the priority queue
        :return: None
        """
        # list is stored by reference, so updating will update all refs
        node = [priority, next(self.counter), vertex]
        self.locator[vertex.id] = node
        heapq.heappush(self.data, node)

    def pop(self) -> Tuple[float, Vertex]:
        """
        Remove and return the (priority, vertex) tuple with lowest priority key
        :return: (priority, vertex) tuple where priority is key,
        and vertex is Vertex object stored in priority queue
        """
        vertex = None
        while vertex is None:
            # keep popping until we have valid entry
            priority, count, vertex = heapq.heappop(self.data)
        del self.locator[vertex.id]            # remove from locator dict
        vertex.visited = True                  # indicate that this vertex was visited
        while len(self.data) > 0 and self.data[0][2] is None:
            heapq.heappop(self.data)          # delete trailing Nones
        return priority, vertex

    def update(self, new_priority: float, vertex: Vertex) -> None:
        """
        Update given Vertex object in the priority queue to have new priority
        :param new_priority: new priority on which to order vertex
        :param vertex: Vertex object for which priority is to be updated
        :return: None
        """
        node = self.locator.pop(vertex.id)      # delete from dictionary
        node[-1] = None                         # invalidate old node
        self.push(new_priority, vertex)         # push new node
