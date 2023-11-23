import random
import matplotlib.pyplot as plt

def create_random_directed_graph(vertices, edge_probability=0.5):
    """
    Create a random directed graph.

    Parameters:
    vertices (list): A list of vertices.
    edge_probability (float, optional): Probability of an edge creation. Defaults to 0.5.

    Returns:
    set: A set of directed edges represented as tuples (vertex1, vertex2).
    """
    edges = set()
    for i in range(len(vertices)):
        for j in range(len(vertices)):
            # Avoid self-loops and add edges based on probability
            if i != j and random.random() < edge_probability:
                edges.add((vertices[i], vertices[j]))
    return edges

def dfs_cycle(vertex, graph, start_vertex, path, visited, cycles):
    """
    Perform a DFS to detect cycles in a graph.

    Parameters:
    vertex: The current vertex in DFS.
    graph: The graph as a set of edges.
    start_vertex: The starting vertex for DFS.
    path: The current path (list of vertices) in DFS.
    visited: Set of visited vertices.
    cycles: Set to store detected cycles.

    Returns:
    None: Modifies the 'cycles' set in place.
    """
    # Check if we have returned to the start vertex and found a cycle
    if vertex == start_vertex and len(path) > 1:
        cycles.add(tuple(path))
        return

    # Skip if vertex is already visited
    if vertex in visited:
        return

    # Add current vertex to visited set and path
    visited.add(vertex)
    path.append(vertex)

    # Explore neighbors
    for _, neighbor in filter(lambda edge: edge[0] == vertex, graph):
        dfs_cycle(neighbor, graph, start_vertex, path.copy(), visited.copy(), cycles)

def break_cycles(graph):
    """
    Detect and break cycles in a directed graph to make it acyclic.

    Parameters:
    graph: The graph as a set of edges.

    Returns:
    int: The number of cycles broken.
    """
    cycles = set()
    # Detect all cycles
    for vertex, _ in graph:
        dfs_cycle(vertex, graph, vertex, [], set(), cycles)

    # If no cycles, return 0
    if not cycles:
        return 0

    # Remove a random edge from a random cycle
    cycle = random.choice(list(cycles))
    cycle_edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
    random_edge = random.choice(cycle_edges)
    graph.remove(random_edge)

    # Recursive call to remove additional cycles
    return 1 + break_cycles(graph)