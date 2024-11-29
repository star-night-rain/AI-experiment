### Usage

#### Compile

To compile the project:

```
make
```

#### Run

To run the project:

```
[1]exe -s [source_city] -t [target_city] -m [method] -h [heuristic_function]
```

1.**source_city**: departure city

2.**target_city**: destination city

3.**method**: search algorithm(e.g.,bfs,dfs,a_star)

4.**heuristic_function**: heuristic function(e.g.,L0,L1,L2)

#### example

```
./run -s Iasi  -t Rimnicu -m a_star -h L2
```
