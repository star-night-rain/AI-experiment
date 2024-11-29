#include "Graph.h"

Graph::Graph()
{
    printf("begin load graph...\n");
    n = 0;
    m = 0;

    string folder = "./data";
    string name_file = folder + "/cityname.txt";
    string position_file = folder + "/citposi.txt";
    string info_file = folder + "/cityinfo.txt";

    fstream fin;
    fin.open(name_file);
    string name;
    while (fin >> name)
        get_vertex_id(name);
    fin.close();

    vertex_position.resize(n);
    fin.open(position_file);
    int x, y;
    while (fin >> name >> x >> y)
    {
        int v = get_vertex_id(name);
        vertex_position[v] = {x, y};
    }
    fin.close();

    edges.resize(n);
    fin.open(info_file);
    int nums;
    while (fin >> name >> nums)
    {
        m += 2 * nums;
        int v = get_vertex_id(name);
        while (nums--)
        {
            string neighbor;
            int distance;
            fin >> neighbor >> distance;
            int u = get_vertex_id(neighbor);
            edges[v].emplace_back(u, distance);
            edges[u].emplace_back(v, distance);
        }
    }
    fin.close();

    printf("n:%d,m:%d\n", n, m);
}

void Graph::search(string source_city, string target_city, string method, string type)
{
    dist.resize(n, inf);
    st.resize(n, false);
    prev.resize(n, -1);

    int s = get_vertex_id(source_city);
    int t = get_vertex_id(target_city);

    auto start_time = high_resolution_clock::now();
    if (method == "bfs")
        bfs(s, t);
    else if (method == "dfs")
    {
        dist[s] = 0;
        dfs(s, t);
    }
    else if (method == "a_star")
    {
        get_h(s, type);
        a_star(s, t);
    }

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end_time - start_time);
    printf("source city:%s,target city:%s\n", source_city.c_str(), target_city.c_str());
    if (method == "a_star")
        printf("heuristic function type:%s\n", type.c_str());
    printf("the runtime of the %s method is:%dms\n",
           method.c_str(), duration.count());

    report_result(t);
}

void Graph::bfs(int s, int t)
{
    queue<int> que;
    que.push(s);
    dist[s] = 0;
    st[s] = true;
    while (!que.empty())
    {
        int v = que.front();
        que.pop();
        for (auto &[u, w] : edges[v])
            if (!st[u])
            {
                st[u] = true;
                dist[u] = dist[v] + w;
                prev[u] = v;
                que.push(u);
            }
    }
}

bool Graph::dfs(int s, int t)
{
    st[s] = true;
    if (s == t)
        return true;
    for (auto &[u, w] : edges[s])
    {
        if (st[u])
            continue;
        dist[u] = dist[s] + w;
        prev[u] = s;
        if (dfs(u, t))
            return true;
        prev[u] = -1;
        dist[u] = inf;
    }
    return false;
}

void Graph::a_star(int s, int t)
{
    priority_queue<PDI, vector<PDI>, greater<PDI>> heap;
    heap.push({h[s], s});
    dist[s] = 0;
    while (!heap.empty())
    {
        auto [d, v] = heap.top();
        heap.pop();
        if (st[v])
            continue;
        st[v] = true;
        d -= h[v];
        for (auto &[u, w] : edges[v])
            if (d + w < dist[u])
            {
                dist[u] = d + w;
                prev[u] = v;
                heap.push({dist[u] + h[u], u});
            }
    }
}

void Graph::get_h(int s, string type)
{
    if (type == "L0")
        h.resize(n, 0);
    else
    {
        h.resize(n, 0);
        auto &[x1, y1] = vertex_position[s];
        for (int i = 0; i < n; i++)
        {
            auto &[x2, y2] = vertex_position[i];
            if (type == "L1")
                h[i] = abs(x1 - x2) + abs(y1 - y2);
            else if (type == "L2")
                h[i] = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
        }
    }
}

int Graph::get_vertex_id(string name)
{
    if (!vertex_id.count(name))
    {
        vertex_id[name] = n;
        vertex_map[n] = name;
        n++;
    }
    return vertex_id[name];
}

bool Graph::check_vertex_existence(string name)
{
    if (!vertex_id.count(name))
        return false;
    else
        return true;
}

void Graph::report_city_name()
{
    printf("[");
    for (auto it = vertex_id.begin(); it != vertex_id.end(); ++it)
        printf("%s ", it->first.c_str());
    printf("]\n");
}

void Graph::report_result(int t)
{
    vector<int> paths;
    paths.reserve(n);
    int v = t;
    while (~v)
    {
        paths.emplace_back(v);
        v = prev[v];
    }
    reverse(paths.begin(), paths.end());

    printf("the results are as follows:\n");
    printf("the number of visited vertices:%d\n", paths.size());
    printf("path distance:%.2lf\n", dist[t]);
    printf("path: ");
    for (int i = 0; i < paths.size(); ++i)
    {
        if (i)
            printf(" -> ");
        printf("%s", vertex_map[paths[i]].c_str());
    }
}