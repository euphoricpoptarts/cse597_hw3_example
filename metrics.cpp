#include "csr_graph.h"
#include <vector>

double min_density(csr_graph& g, std::vector<int>& cluster_idx){
    int n = g.t_vtx;
    std::vector<int> cluster_volume(n, 0);
    std::vector<int> cluster_coverage(n, 0);
    for(int v = 0; v < n; v++){
        int cv = cluster_idx[v];
        int degree = g.row_map[v+1] - g.row_map[v];
        cluster_volume[cv] += degree;
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
        if(cluster_volume[c] > 0){
            double density = static_cast<double>(cluster_coverage[c]) / static_cast<double>(cluster_volume[c]);
            if(density < min){
                min = density;
            }
        }
    }
    return min;
}

double avg_isolated_conductance(csr_graph& g, std::vector<int>& cluster_idx){
    int n = g.t_vtx;
    std::vector<int> cluster_volume(n, 0);
    std::vector<int> cluster_cut(n, 0);
    std::vector<int> nonempty_clusters(n, 0);
    for(int v = 0; v < n; v++){
        int cv = cluster_idx[v];
        int degree = g.row_map[v+1] - g.row_map[v];
        cluster_volume[cv] += degree;
        nonempty_clusters[cv]++;
        for(int j = g.row_map[v]; j < g.row_map[v+1]; j++){
            int u = g.entries[j];
            int cu = cluster_idx[u];
            if(cv != cu){
                cluster_cut[cv]++;
            }
        }
    }
    double min = 1;
    double t_clusters = 0;
    double conductance_sum = 0;
    for(int c = 0; c < n; c++){
        // assume that the density of any cluster with no edges is 1
        if(nonempty_clusters[c] > 0){
            t_clusters += 1.0;
            // multiply by 2 to account for other side of each cut edge
            double cut = cluster_cut[c] * 2;
            int min_vol = cluster_volume[c];
            if(g.nnz - min_vol < min_vol) min_vol = g.nnz - min_vol;
            // prevent divide by zero
            if(min_vol == 0) min_vol = 1;
            double conductance = cut / static_cast<double>(min_vol);
            conductance_sum += conductance;
        }
    }
    return conductance_sum / t_clusters;
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
    std::vector<int> cluster_volume(n, 0);
    for(int v = 0; v < n; v++){
        int cv = cluster_idx[v];
        int degree = g.row_map[v+1] - g.row_map[v];
        cluster_volume[cv] += degree;
    }
    double inv_nnz = 1.0 / static_cast<double>(g.nnz);
    double mod = coverage(g, cluster_idx);
    double penalty = 0;
    for(int c = 0; c < n; c++){
        double volume = cluster_volume[c];
        penalty += volume*volume;
    }
    mod -= penalty*inv_nnz*inv_nnz;
    return mod;
}