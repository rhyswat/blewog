# Topological sorting on a graph

class Graph:
    def __init__(self):
        '''
        This implementation maps the vertex N to the set of vertices with in-edges into N.
        For example this models the relation 'N depends-on {these other things}'.
        '''
        self.graph = dict()
        self.graph[7] = set()
        self.graph[5] = set()
        self.graph[3] = set()
        self.graph[11] = set([7, 5])
        self.graph[8] = set([7, 3])
        self.graph[2] = set([11])
        self.graph[9] =  set([11, 8])
        self.graph[10] = set([3, 11])
        
    def nodes_with_zero_in_degree(self):
        '''
        Get the vertices in this graph that have zero in-degree.
        If the graph is a DAG then this collection will be non-empty.
        '''
        return [x for x in self.graph if len(self.graph[x]) == 0]

    def out_degree(self, n):
        '''Get the out-degree of a vertex n in this graph.'''
        return len([x for x in self.graph if n in self.graph[x]])

    def prune(self, n):
        '''
        Removes all edges starting at n, returning those
        vertices that were connected to n.
        Those vertices could still have in-edges from other vertices.
        '''
        pruned = set()
        for m, i in self.graph.items():
            if n in i:
                i.remove(n)
                if len(i) == 0:
                    pruned.add(m)
        return pruned

    def has_edges(self):
        '''Test whether this graph has edges.'''
        return any(len(s) > 0 for s in self.graph.values())

class NodeSource:
    '''Book-keeping class for storing and yielding nodes.'''
    def __init__(self, graph=None):
        self.items = []

    def pop(self):
        return self.items.pop(0)

    def add(self, nodes):
        self.items.extend(nodes)

    def is_empty(self):
        return len(self.items) == 0

    def __repr__(self):
        return repr(self.items)

class SmallestFirstSource(NodeSource):
    '''Yields nodes in smallest value first order.'''
    def __init__(self):
        super().__init__()

    def pop(self):
        return self.items.pop(0)

    def add(self, nodes):
        self.items.extend(nodes)
        self.items.sort()

class LargestFirstSource(NodeSource):
    '''Yields nodes in largest value first order.'''
    def __init__(self):
        super().__init__()

    def pop(self):
        return self.items.pop()

    def add(self, nodes):
        self.items.extend(nodes)
        self.items.sort()

class LargestOutDegreeSource(NodeSource):
    '''
    Yields nodes in largest out degree order.
    This processes nodes with many dependencies first.
    '''
    def __init__(self, graph):
        super().__init__()
        self.graph = graph

    def pop(self):
        return self.items.pop()

    def add(self, nodes):
        self.items.extend(nodes)
        u = lambda x : self.graph.out_degree(x)
        self.items.sort(key=u)

# Main algorithm
def tsort(graph, src, verbose=False):
    '''
    Implements topological sorting.
    
    ** This algorithm is destructive **
    It suucessively prunes the graph removing edges as it goes.
    
    Takes a Graph and a NodeSource object.

    Returns a list of nodes in topological order, which depends
    on the node source's behaviour.
    
    Raises a ValueError if the graph is cyclic.
    '''
    sequence = list()
    s.add(graph.nodes_with_zero_in_degree())
    while not s.is_empty():
        n = s.pop()
        sequence.append(n)
        s.add(graph.prune(n))

        if verbose:
            print(' N:', repr(n))
            print(' S:', repr(sequence))
            print(' R:', repr(s))
            print()
        
    if graph.has_edges():
        raise ValueError('Graph has cycles, cannot sort topologically')
    return sequence

if __name__ == '__main__':
    g = Graph()
    s = LargestOutDegreeSource(g)
    t = tsort(g, s)
    print(t)
