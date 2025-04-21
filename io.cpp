#include <fstream>
#include <iostream>
#include "csr_graph.h"

int fast_atoi( const char*& str )
{
    int val = 0;
    while(isdigit(*str)) {
        val = val*10 + static_cast<int>(*str - '0');
        str++;
    }
    return val;
}

void next_line(const char*& str){
    while(*str != '\n') str++;
    str++;
}

csr_graph load_metis_graph(const char *fname) {
    std::ifstream infp(fname, std::ios::binary);
    csr_graph g;
    if (!infp.is_open()) {
        std::cerr << "FATAL ERROR: Could not open metis graph file " << fname << std::endl;
        g.error = true;
        return g;
    }
    infp.seekg(0, std::ios::end);
    size_t sz = infp.tellg();
    std::cout << "Reading " << sz << " bytes from " << fname << std::endl;
    infp.seekg(0, std::ios::beg);
    //1 for extra newline if needed
    char* s = new char[sz + 1];
    infp.read(s, sz);
    infp.close();
    std::cout << "Finished reading bytes from " << fname << std::endl;
    //append an endline to end of file in case one doesn't exist
    //needed to prevent parser from overshooting end of buffer
    if(s[sz - 1] != '\n'){
        s[sz] = '\n';
        sz++;
    }
    const char* f = s;
    const char* fmax = s + sz;
    size_t header[4] = {0, 0, 0, 0};
    //ignore commented lines
    while(*f == '%') next_line(f);
    while(!isdigit(*f)) f++;
    //read header data
    for(int i = 0; i < 4; i++){
        header[i] = fast_atoi(f);
        while(!isdigit(*f)){
            if(*f == '\n'){
                i = 4;
                f++;
                break;
            }
            f++;
        }
    }
    int n = header[0];
    int m = header[1];
    int fmt = header[2];
    int ncon = header[3];
    bool has_ew = ((fmt % 10) == 1);
    if(fmt != 0 && fmt != 1){
        std::cerr << "FATAL ERROR: Unsupported format flags " << fmt << std::endl;
        std::cerr << "Graph parser does not currently support vertex weights" << std::endl;
        g.error = true;
        return g;
    }
    if(ncon != 0){
        std::cerr << "FATAL ERROR: Unsupported ncon " << ncon << std::endl;
        std::cerr << "Graph parser does not currently support vertex weights" << std::endl;
        g.error = true;
        return g;
    }
    std::vector<int> entries(m*2);
    std::vector<int> row_map(n + 1);
    row_map[0] = 0;
    int edges_read = 0;
    int rows_read = 0;
    //read edge information
    while(f < fmax){
        //increment past whitespace
        while(f < fmax && !isdigit(*f)){
            //ignore commented lines
            if(*f == '%'){
                next_line(f);
                continue;
            }
            if(*f == '\n'){
                //ignore extra trailing newlines
                if(rows_read < n) row_map[++rows_read] = edges_read;
            }
            f++;
        }
        if(f >= fmax) break;
        //fast_atoi also increments past numeric chars
        int edge_info = fast_atoi(f);
        //subtract 1 to convert to 0-indexed
        entries[edges_read] = edge_info - 1;
        edges_read++;
    }
    delete[] s;
    if(rows_read != n || edges_read != 2*m){
        std::cerr << "FATAL ERROR: Mismatch between expected and actual line/nonzero count in metis file" << std::endl;
        std::cerr << "Read " << rows_read << " lines and " << edges_read << " nonzeros" << std::endl;
        std::cerr << "Lines expected: " << n << "; Nonzeros expected: " << m*2 << std::endl;
        g.error = true;
        return g;
    }
    std::cout << "Processed bytes into graph" << std::endl;
    g.row_map = row_map;
    g.entries = entries;
    g.t_vtx = n;
    g.nnz = 2*m;
    g.error = false;
    return g;
}