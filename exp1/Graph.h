#include <iostream>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <getopt.h>
#include <set>
#include <queue>
#include <cmath>
#include <chrono>
#include <algorithm>
using namespace std;
using namespace std::chrono;

#define inf 0x3f3f3f3f
typedef pair<int, int> PII;
typedef pair<double, int> PDI;

class Graph
{
public:
    // the number of vertices and edges
    int n, m;

    // vertex name -> vertex id
    unordered_map<string, int> vertex_id;
    // vertex id -> vertex name
    unordered_map<int, string> vertex_map;

    // the position of each vertex
    // data format:(x,y)
    vector<PII> vertex_position;

    // the neighbors of each vertex
    // data format:(neighbor,weight)
    vector<vector<PII>> edges;

    // heuristic function
    vector<double> h;

    // search
    vector<double> dist;
    vector<bool> st;
    vector<int> prev;

    Graph();

    void search(string source_city, string target_city, string method, string type);

    void bfs(int s, int t);
    bool dfs(int s, int t);
    void a_star(int s, int t);

    void get_h(int s, string type);
    int get_vertex_id(string name);

    // check the existence of given vertex
    bool check_vertex_existence(string name);

    void report_city_name();

    // report results
    void report_result(int t);
};