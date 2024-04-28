## Graphs
### Implement BFS
```
 // Function to perform Breadth First Search on a graph
    // represented using adjacency list
    void bfs(int startNode)
    {
        // Create a queue for BFS
        Queue<Integer> queue = new LinkedList<>();
        boolean[] visited = new boolean[vertices];

        // Mark the current node as visited and enqueue it
        visited[startNode] = true;
        queue.add(startNode);

        // Iterate over the queue
        while (!queue.isEmpty()) {
            // Dequeue a vertex from queue and print it
            int currentNode = queue.poll();
            System.out.print(currentNode + " ");

            // Get all adjacent vertices of the dequeued
            // vertex currentNode If an adjacent has not
            // been visited, then mark it visited and
            // enqueue it
            for (int neighbor : adjList[currentNode]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    queue.add(neighbor);
                }
            }
        }
    }
```
### Implement Breadth First Search
```
    // A function used by DFS
    void DFSUtil(int v, boolean visited[])
    {
        // Mark the current node as visited and print it
        visited[v] = true;
        System.out.print(v + " ");
 
        // Recur for all the vertices adjacent to this
        // vertex
        Iterator<Integer> i = adj[v].listIterator();
        while (i.hasNext()) {
            int n = i.next();
            if (!visited[n])
                DFSUtil(n, visited);
        }
    }
 
    // The function to do DFS traversal.
    // It uses recursive DFSUtil()
    void DFS(int v)
    {
        // Mark all the vertices as
        // not visited(set as
        // false by default in java)
        boolean visited[] = new boolean[V];
 
        // Call the recursive helper
        // function to print DFS
        // traversal
        DFSUtil(v, visited);
    }
```
### Cycle detection in a Directed Graph
```
// Function to check if cycle exists
    private boolean isCyclicUtil(int i, boolean[] visited,
                                 boolean[] recStack)
    {
 
        // Mark the current node as visited and
        // part of recursion stack
        visited[i] = true;
        recStack[i] = true;
        List<Integer> children = adj.get(i);
 
        for (Integer c : children) {
            if (recStack[c]) {
                return True
            }
            if (!visited[c]) {
                if (isCyclicUtil(c, visited, recStack))
                return true;
            }
            
        }
        recStack[i] = false;
        return false;
    }

 
    // Returns true if the graph contains a
    // cycle, else false.
    private boolean isCyclic()
    {
        // Mark all the vertices as not visited and
        // not part of recursion stack
        boolean[] visited = new boolean[V];
        boolean[] recStack = new boolean[V];
 
        // Call the recursive helper function to
        // detect cycle in different DFS trees
        for (int i = 0; i < V; i++)
            if (isCyclicUtil(i, visited, recStack))
                return true;
 
        return false;
    }
```
### Cycle detection in a UnDirected Graph
```
Boolean isCyclicUtil(int v, Boolean visited[],
                         int parent)
    {
        // Mark the current node as visited
        visited[v] = true;
        Integer i;
 
        // Recur for all the vertices
        // adjacent to this vertex
        Iterator<Integer> it = adj[v].iterator();
        while (it.hasNext()) {
            i = it.next();
 
            // If an adjacent is not
            // visited, then recur for that
            // adjacent
            if (!visited[i]) {
                if (isCyclicUtil(i, visited, v))
                    return true;
            }
 
            // If an adjacent is visited
            // and not parent of current
            // vertex, then there is a cycle.
            else if (i != parent)
                return true;
        }
        return false;
    }
```
### Mother vertex in a undirected graph
```
// A recursive function to print DFS starting from v
    static void DFSUtil(ArrayList<ArrayList<Integer> > g,
                        int v, boolean[] visited)
    {
        // Mark the current node as
        // visited and print it
        visited[v] = true;
 
        // Recur for all the vertices
        // adjacent to this vertex
        for (int x : g.get(v)) {
            if (!visited[x]) {
                DFSUtil(g, x, visited);
            }
        }
    }
 
    // Returns a mother vertex if exists.
    // Otherwise returns -1
    static int
    motherVertex(ArrayList<ArrayList<Integer> > g, int V)
    {
 
        // visited[] is used for DFS. Initially
        // all are initialized as not visited
        boolean[] visited = new boolean[V];
 
        // To store last finished vertex
        // (or mother vertex)
        int v = -1;
 
        for (int i = 0; i < V; i++) {
            if (!visited[i]) {
                DFSUtil(g, i, visited);
                v = i;
            }
        }
 
        // If there exist mother vertex (or vertices)
        // in given graph, then v must be one
        // (or one of them)
 
        // Now check if v is actually a mother
        // vertex (or graph has a mother vertex).
        // We basically check if every vertex
        // is reachable from v or not.
 
        // Reset all values in visited[] as false
        // and do DFS beginning from v to check
        // if all vertices are reachable from
        // it or not.
        boolean[] check = new boolean[V];
        DFSUtil(g, v, check);
        for (boolean val : check) {
            if (!val) {
                return -1;
            }
        }
        return v;
    }
```
### Check if a Directed Graph is Tree or Not
An undirected graph is a tree if it has the following properties
- There is no cycle. 
- The graph is connected.
```
    // A recursive function that uses visited[] and parent 
    // to detect cycle in subgraph reachable from vertex v. 
    boolean isCyclicUtil(int v, boolean visited[], int parent) 
    { 
        // Mark the current node as visited 
        visited[v] = true; 
        Integer i; 
 
        // Recur for all the vertices adjacent to this vertex 
        Iterator<Integer> it = adj[v].iterator(); 
        while (it.hasNext()) 
        { 
            i = it.next(); 
 
            // If an adjacent is not visited, then recur for 
            // that adjacent 
            if (!visited[i]) 
            { 
                if (isCyclicUtil(i, visited, v)) 
                    return true; 
            } 
 
            // If an adjacent is visited and not parent of 
            // current vertex, then there is a cycle. 
            else if (i != parent) 
            return true; 
        } 
        return false; 
    } 
 
    // Returns true if the graph is a tree, else false. 
    boolean isTree() 
    { 
        // Mark all the vertices as not visited and not part 
        // of recursion stack 
        boolean visited[] = new boolean[V]; 
        for (int i = 0; i < V; i++) 
            visited[i] = false; 
 
        // The call to isCyclicUtil serves multiple purposes 
        // It returns true if graph reachable from vertex 0 
        // is cyclic. It also marks all vertices reachable 
        // from 0. 
        if (isCyclicUtil(0, visited, -1)) 
            return false; 
 
        // If we find a vertex which is not reachable from 0 
        // (not marked by isCyclicUtil(), then we return false 
        for (int u = 0; u < V; u++) 
            if (!visited[u]) 
                return false; 
 
        return true; 
    } 
```
### Shortest distance between two vertices
```
public static int findMin(Graph g, int source, int destination) {
        if (source == destination) {
            return 0;
        }

        int result = 0;
        int num_of_vertices = g.vertices;

        //Boolean Array to hold the history of visited nodes (by default-false)
        //Make a node visited whenever you enqueue it into queue
        boolean[] visited = new boolean[num_of_vertices];

        //For keeping track of distance of current_node from source
        int[] distance = new int[num_of_vertices];

        //Create Queue for Breadth First Traversal and enqueue source in it
        Queue < Integer > queue = new Queue < Integer > (num_of_vertices);

        queue.enqueue(source);
        visited[source] = true;

        //Traverse while queue is not empty
        while (!queue.isEmpty()) {

            //Dequeue a vertex/node from queue and add it to result
            int current_node = queue.dequeue();

            //Get adjacent vertices to the current_node from the array,
            //and if they are not already visited then enqueue them in the Queue
            //and also update their distance from source by adding 1 in current_nodes's distance
            DoublyLinkedList < Integer > .Node temp = null;
            if (g.adjacencyList[current_node] != null)
                temp = g.adjacencyList[current_node].headNode;

            while (temp != null) {

                if (!visited[temp.data]) {
                    queue.enqueue(temp.data);
                    visited[temp.data] = true;
                    distance[temp.data] = distance[current_node] + 1;
                }
                if (temp.data == destination) {
                    return distance[destination];
                }
                temp = temp.nextNode;
            }
        } //end of while
        return -1;
    }
```
## Topological sort
```
static void
    topologicalSortUtil(int v, List<List<Integer> > adj,
                        boolean[] visited,
                        Stack<Integer> stack)
    {
        // Mark the current node as visited
        visited[v] = true;

        // Recur for all adjacent vertices
        for (int i : adj.get(v)) {
            if (!visited[i])
                topologicalSortUtil(i, adj, visited, stack);
        }

        // Push current vertex to stack which stores the
        // result
        stack.push(v);
    }

    // Function to perform Topological Sort
    static void topologicalSort(List<List<Integer> > adj,
                                int V)
    {
        // Stack to store the result
        Stack<Integer> stack = new Stack<>();
        boolean[] visited = new boolean[V];

        // Call the recursive helper function to store
        // Topological Sort starting from all vertices one
        // by one
        for (int i = 0; i < V; i++) {
            if (!visited[i])
                topologicalSortUtil(i, adj, visited, stack);
        }

        // Print contents of stack
        System.out.print(
            "Topological sorting of the graph: ");
        while (!stack.empty()) {
            System.out.print(stack.pop() + " ");
        }
    }
```
### Alien Dictionary
### 