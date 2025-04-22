#include "io.cpp"
#include "csr_graph.h"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <numeric>
#include <random>
#include <chrono>
#include <queue>

double min_density(csr_graph& g, std::vector<int>& cluster_idx){
    int n = g.t_vtx;
    std::vector<int> cluster_degree(n, 0);
    std::vector<int> cluster_coverage(n, 0);
    for(int v = 0; v < n; v++){
        int cv = cluster_idx[v];
        int degree = g.row_map[v+1] - g.row_map[v];
        cluster_degree[cv] += degree;
        for(int j = g.row_map[v]; j < g.row_map[v+1]; j++){
            int u = g.entries[j];
            int cu = cluster_idx[u];
            if(cv == cu){
                cluster_coverage[cv]++;
            }
        }
    }
    double min = 1;
    for(int c = 0; c < n; c++){
        // assume that the density of any cluster with no edges is 1
        if(cluster_degree[c] > 0){
            double density = static_cast<double>(cluster_coverage[c]) / static_cast<double>(cluster_degree[c]);
            if(density < min){
                min = density;
            }
        }
    }
    return min;
}

double coverage(csr_graph& g, std::vector<int>& cluster_idx){
    int n = g.t_vtx;
    int coverage = 0;
    for(int v = 0; v < n; v++){
        int cv = cluster_idx[v];
        for(int j = g.row_map[v]; j < g.row_map[v+1]; j++){
            int u = g.entries[j];
            int cu = cluster_idx[u];
            if(cv == cu){
                coverage++;
            }
        }
    }
    double inv_nnz = 1.0 / static_cast<double>(g.nnz);
    return static_cast<double>(coverage) * inv_nnz;
}

double modularity(csr_graph& g, std::vector<int>& cluster_idx){
    int n = g.t_vtx;
    std::vector<int> cluster_degree(n, 0);
    for(int v = 0; v < n; v++){
        int cv = cluster_idx[v];
        int degree = g.row_map[v+1] - g.row_map[v];
        cluster_degree[cv] += degree;
    }
    double inv_nnz = 1.0 / static_cast<double>(g.nnz);
    double mod = coverage(g, cluster_idx);
    double penalty = 0;
    for(int c = 0; c < n; c++){
        double degree = cluster_degree[c];
        penalty += degree*degree;
    }
    mod -= penalty*inv_nnz*inv_nnz;
    return mod;
}

// most basic implementation of label propagation
std::vector<int> label_propagation_v1(csr_graph& g){
    int n = g.t_vtx;
    std::vector<int> cluster_idx(n);
    // initializes cluster_idx as 0, 1, 2, 3, 4, ... (n - 2), (n - 1)
    std::iota(cluster_idx.begin(), cluster_idx.end(), 0);
    // do some number of iterations
    for(int lp_iteration = 0; lp_iteration < 100; lp_iteration++){
        auto start = std::chrono::high_resolution_clock::now();
        // iterate over ALL vertices
        for(int v = 0; v < n; v++){
            // hashmap to hold each adjacent cluster (key) and sum of edges in that cluster (value)
            std::unordered_map<int, int> conn_strength;
            // fill hashmap
            for(int j = g.row_map[v]; j < g.row_map[v+1]; j++){
                int u = g.entries[j];
                int cu = cluster_idx[u];
                // add to hashtable
                if(!conn_strength.contains(cu)){
                    conn_strength[cu] = 0;
                }
                // increment strength
                conn_strength[cu]++;
            }
            int max_val = 0;
            int argmax = n + 1;
            // iterate over hashmap
            // find argmax as most connected adjacent cluster
            for(auto& it : conn_strength){
                int key = it.first;
                int val = it.second;
                // minimum label heuristic to break ties
                if(val > max_val || (val == max_val && key < argmax)){
                    argmax = key;
                    max_val = val;
                }
            }
            // update cluster idx
            if(max_val > 0){
                cluster_idx[v] = argmax;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Iteration time: " << duration << std::endl;
    }
    return cluster_idx;
}

// v1 + a more efficient datastructure
std::vector<int> label_propagation_v2(csr_graph& g){
    int n = g.t_vtx;
    std::vector<int> cluster_idx(n);
    // initializes cluster_idx as 0, 1, 2, 3, 4, ... (n - 2), (n - 1)
    std::iota(cluster_idx.begin(), cluster_idx.end(), 0);
    // more effective (though more memory intensive) datastructure
    // contains an entry for every possible cluster id
    std::vector<int> conn_strength(n, 0);
    // conn keys holds a list of the current nonzero entries in conn_strength
    std::vector<int> conn_keys(n, 0);
    // do some number of iterations
    for(int lp_iteration = 0; lp_iteration < 100; lp_iteration++){
        auto start = std::chrono::high_resolution_clock::now();
        // iterate over ALL vertices
        for(int v = 0; v < n; v++){
            int adj_clusters = 0;
            for(int j = g.row_map[v]; j < g.row_map[v+1]; j++){
                int u = g.entries[j];
                int cu = cluster_idx[u];
                // update nonzero entries list
                if(conn_strength[cu] == 0){
                    conn_keys[adj_clusters++] = cu;
                }
                // increment strength
                conn_strength[cu]++;
            }
            int max_val = 0;
            int argmax = n + 1;
            // iterate over nonzero entries
            // find argmax as most connected adjacent cluster
            for(int cx = 0; cx < adj_clusters; cx++){
                int key = conn_keys[cx];
                int val = conn_strength[key];
                // reset to zero for next vertex
                conn_strength[key] = 0;
                // minimum label heuristic to break ties
                if(val > max_val || (val == max_val && key < argmax)){
                    argmax = key;
                    max_val = val;
                }
            }
            if(max_val > 0){
                cluster_idx[v] = argmax;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Iteration time: " << duration << std::endl;
    }
    return cluster_idx;
}

// v2 + vertex pruning
std::vector<int> label_propagation_v3(csr_graph& g){
    int n = g.t_vtx;
    std::vector<int> cluster_idx(n);
    // initializes cluster_idx as 0, 1, 2, 3, 4, ... (n - 2), (n - 1)
    std::iota(cluster_idx.begin(), cluster_idx.end(), 0);
    // more effective (though more memory intensive) datastructure
    // contains an entry for every possible cluster id
    std::vector<int> conn_strength(n, 0);
    // conn keys holds a list of the current nonzero entries in conn_strength
    std::vector<int> conn_keys(n, 0);
    // tracks the last iteration in which an adjacent vertex was modified
    std::vector<char> modified(n, -1);
    auto start = std::chrono::high_resolution_clock::now();
    // iterate until convergance (no vertices change cluster in an iteration)
    for(int lp_iteration = 0; ; lp_iteration++){
        int t_modified = 0;
        for(int v = 0; v < n; v++){
            // if no adjacent vertex was modified in or after the last iteration
            // do not process this vertex
            if(modified[v] + 1 < lp_iteration) continue;
            int adj_clusters = 0;
            for(int j = g.row_map[v]; j < g.row_map[v+1]; j++){
                int u = g.entries[j];
                int cu = cluster_idx[u];
                // update nonzero entries list
                if(conn_strength[cu] == 0){
                    conn_keys[adj_clusters++] = cu;
                }
                // increment strength
                conn_strength[cu]++;
            }
            int max_val = 0;
            int argmax = n + 1;
            // iterate over nonzero entries
            // find argmax as most connected adjacent cluster
            for(int cx = 0; cx < adj_clusters; cx++){
                int key = conn_keys[cx];
                int val = conn_strength[key];
                // reset to zero for next vertex
                conn_strength[key] = 0;
                // minimum label heuristic to break ties
                if(val > max_val || (val == max_val && key < argmax)){
                    argmax = key;
                    max_val = val;
                }
            }
            if(max_val > 0){
                if(cluster_idx[v] != argmax){
                    cluster_idx[v] = argmax;
                    // mark all adjacent vertices with current iteration
                    for(int j = g.row_map[v]; j < g.row_map[v+1]; j++){
                        int u = g.entries[j];
                        modified[u] = lp_iteration;
                    }
                    t_modified++;
                }
            }
        }
        // no vertices were modified, then stop
        if(t_modified == 0) break;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Total time: " << duration << std::endl;
    return cluster_idx;
}

// v3 + random vertex traversal order
// what effect do you think the randomization will have on the runtime?
// what about the modularity?
std::vector<int> label_propagation_v4(csr_graph& g){
    int n = g.t_vtx;
    std::vector<int> cluster_idx(n);
    std::vector<int> order(n);
    // initializes cluster_idx and order as 0, 1, 2, 3, 4, ... (n - 2), (n - 1)
    std::iota(cluster_idx.begin(), cluster_idx.end(), 0);
    std::iota(order.begin(), order.end(), 0);
    // more effective (though more memory intensive) datastructure
    // contains an entry for every possible cluster id
    std::vector<int> conn_strength(n, 0);
    // conn keys holds a list of the current nonzero entries in conn_strength
    std::vector<int> conn_keys(n, 0);
    // tracks the last iteration in which an adjacent vertex was modified
    std::vector<char> modified(n, -1);
    // shuffle the vertex ordering randomly
    std::shuffle(order.begin(), order.end(), std::mt19937{std::random_device{}()});
    // iterate until convergance (no vertices change cluster in an iteration)
    for(int lp_iteration = 0; ; lp_iteration++){
        auto start = std::chrono::high_resolution_clock::now();
        int t_modified = 0;
        for(int i = 0; i < n; i++){
            int v = order[i];
            // if no adjacent vertex was modified in or after the last iteration
            // do not process this vertex
            if(modified[v] + 1 < lp_iteration) continue;
            int adj_clusters = 0;
            for(int j = g.row_map[v]; j < g.row_map[v+1]; j++){
                int u = g.entries[j];
                int cu = cluster_idx[u];
                // update nonzero entries list
                if(conn_strength[cu] == 0){
                    conn_keys[adj_clusters++] = cu;
                }
                // increment strength
                conn_strength[cu]++;
            }
            int max_val = 0;
            int argmax = n + 1;
            // iterate over nonzero entries
            // find argmax as most connected adjacent cluster
            for(int cx = 0; cx < adj_clusters; cx++){
                int key = conn_keys[cx];
                int val = conn_strength[key];
                // reset to zero for next vertex
                conn_strength[key] = 0;
                // minimum label heuristic to break ties
                if(val > max_val || (val == max_val && key < argmax)){
                    argmax = key;
                    max_val = val;
                }
            }
            if(max_val > 0){
                if(cluster_idx[v] != argmax){
                    cluster_idx[v] = argmax;
                    // mark all adjacent vertices with current iteration
                    for(int j = g.row_map[v]; j < g.row_map[v+1]; j++){
                        int u = g.entries[j];
                        modified[u] = lp_iteration;
                    }
                    t_modified++;
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Iteration time: " << duration << std::endl;
        // no vertices were modified, then stop
        if(t_modified == 0) break;
    }
    return cluster_idx;
}

int main(int argc, char** argv){
    if(argc != 2){
        std::cerr << "Incorrect argument count" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <metis filename>" << std::endl;
    }
    csr_graph g = load_metis_graph(argv[1]);
    if(g.error){
        return 1;
    }
    std::vector<int> cluster_idx = label_propagation_v5(g);
    std::cout << "Modularity: " << modularity(g, cluster_idx) << std::endl;
    std::cout << "Coverage: " << coverage(g, cluster_idx) << std::endl;
    std::cout << "Minimum intra-cluster density: " << min_density(g, cluster_idx) << std::endl;
    return 0;
}