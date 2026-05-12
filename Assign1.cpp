#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>
using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); // undirected graph
    }

    // -------- SEQUENTIAL BFS --------
    void sequentialBFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;

        q.push(start);
        visited[start] = true;

        while (!q.empty()) {
            int node = q.front();
            q.pop();

            cout << node << " ";

            for (int neigh : adj[node]) {
                if (!visited[neigh]) {
                    visited[neigh] = true;
                    q.push(neigh);
                }
            }
        }
    }

    // -------- SEQUENTIAL DFS --------
    void sequentialDFS(int start) {
        vector<bool> visited(V, false);
        stack<int> s;

        s.push(start);
        visited[start] = true;

        while (!s.empty()) {
            int node = s.top();
            s.pop();

            cout << node << " ";

            for (int neigh : adj[node]) {
                if (!visited[neigh]) {
                    visited[neigh] = true;
                    s.push(neigh);
                }
            }
        }
    }

    // -------- PARALLEL BFS --------
    void parallelBFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;

        q.push(start);
        visited[start] = true;

        while (!q.empty()) {
            int size = q.size();
            vector<int> level;

            // get current level
            for (int i = 0; i < size; i++) {
                int node = q.front();
                q.pop();
                cout << node << " ";
                level.push_back(node);
            }

            // process level in parallel
#pragma omp parallel for
            for (int i = 0; i < level.size(); i++) {
                int node = level[i];

                for (int neigh : adj[node]) {
#pragma omp critical
                    {
                        if (!visited[neigh]) {
                            visited[neigh] = true;
                            q.push(neigh);
                        }
                    }
                }
            }
        }
    }

    // -------- PARALLEL DFS --------
    void parallelDFS(int start) {
        vector<bool> visited(V, false);
        stack<int> s;

        s.push(start);

#pragma omp parallel
        {
            while (true) {
                int node = -1;

#pragma omp critical
                {
                    if (!s.empty()) {
                        node = s.top();
                        s.pop();
                    }
                }

                if (node == -1) break;

#pragma omp critical
                {
                    if (visited[node]) node = -1;
                    else {
                        visited[node] = true;
                        cout << node << " ";
                    }
                }

                if (node == -1) continue;

                for (int neigh : adj[node]) {
#pragma omp critical
                    s.push(neigh);
                }
            }
        }
    }
};

int main() {
    omp_set_num_threads(4);

    Graph g(6);

    g.addEdge(0,1);
    g.addEdge(0,2);
    g.addEdge(1,3);
    g.addEdge(1,4);
    g.addEdge(2,4);
    g.addEdge(3,5);
    g.addEdge(4,5);

    double start, end;

    // ---- Sequential BFS ----
    cout << "Sequential BFS: ";
    start = omp_get_wtime();
    g.sequentialBFS(0);
    end = omp_get_wtime();
    cout << "\nTime: " << (end - start)*1000 << " ms\n\n";

    // ---- Sequential DFS ----
    cout << "Sequential DFS: ";
    start = omp_get_wtime();
    g.sequentialDFS(0);
    end = omp_get_wtime();
    cout << "\nTime: " << (end - start)*1000 << " ms\n\n";

    // ---- Parallel BFS ----
    cout << "Parallel BFS: ";
    start = omp_get_wtime();
    g.parallelBFS(0);
    end = omp_get_wtime();
    cout << "\nTime: " << (end - start)*1000 << " ms\n\n";

    // ---- Parallel DFS ----
    cout << "Parallel DFS: ";
    start = omp_get_wtime();
    g.parallelDFS(0);
    end = omp_get_wtime();
    cout << "\nTime: " << (end - start)*1000 << " ms\n";

    return 0;
}