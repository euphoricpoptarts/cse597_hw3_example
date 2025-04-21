#pragma once
#include <vector>

struct csr_graph {
    std::vector<int> row_map;
    std::vector<int> entries;
    int t_vtx;
    int nnz;
    bool error;
};