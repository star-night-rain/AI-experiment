#include "Graph.h"

int main(int argc, char *argv[])
{
	Graph graph;

	string source_city = "Arad";
	string target_city = "Bucharest";
	string method = "a_star";
	string type = "L0";
	int option = -1;
	set<string> methods = {"bfs", "dfs", "a_star"};
	set<string> types = {"L0", "L1", "L2"};
	while (-1 != (option = getopt(argc, argv, "s:t:m:h:")))
	{
		if (option == 's')
		{
			if (!graph.check_vertex_existence(optarg))
			{
				printf("please specify the source city -s in ");
				graph.report_city_name();
				return 0;
			}
			source_city = optarg;
		}
		else if (option == 't')
		{
			if (!graph.check_vertex_existence(optarg))
			{
				printf("please specify the target city -t in ");
				graph.report_city_name();
				return 0;
			}
			target_city = optarg;
		}
		else if (option == 'm')
		{
			if (!methods.count(optarg))
			{
				printf("please specify the method -m in [\"bfs\",\"dfs\", \"a_star\"]\n");
				return 0;
			}
			method = optarg;
		}
		else if (option == 'h')
		{
			if (!types.count(string(optarg)))
			{
				printf("please specify the heuristic function type -h in [\"L0\", \"L1\", \"L2\"]\n");
				return 0;
			}
			type = optarg;
		}
	}

	graph.search(source_city, target_city, method, type);
	return 0;
}