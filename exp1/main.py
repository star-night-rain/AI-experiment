import tkinter as tk
import subprocess
import os

city_list = [
    "Arad", "Bucharest", "Craiova", "Drobeta", "Eforie", "Fagaras", "Giurgiu",
    "Hirsova", "Iasi", "Lugoj", "Mehadia", "Neamt", "Oradea", "Pitesti",
    "Rimnicu", "Sibiu", "Timisoara", "Urziceni", "Vaslui", "Zerind"
]

method_list = ['bfs', 'dfs', 'dijkstra', 'a_star(L1)', 'a_star(L2)']


class GUI:

    def __init__(self, master):
        self.master = master
        master.title("Search Algorithm")

        self.source_label = tk.Label(master,
                                     font=("Helvetica", 20, "bold"),
                                     text='source city')
        self.source_label.grid(row=0, column=0, padx=15, pady=15)

        self.source_var = tk.StringVar(master)
        self.source_var.set('Arad')
        self.source_menu = tk.OptionMenu(master, self.source_var, *city_list)
        self.source_menu.config(width=20, font=20)
        self.source_menu.grid(row=0, column=1, padx=15, pady=15)

        self.target_label = tk.Label(master,
                                     font=("Helvetica", 20, "bold"),
                                     text='target city')
        self.target_label.grid(row=1, column=0, padx=15, pady=15)

        self.target_var = tk.StringVar(master)
        self.target_var.set('Bucharest')
        self.target_menu = tk.OptionMenu(master, self.target_var, *city_list)
        self.target_menu.config(width=20, font=20)
        self.target_menu.grid(row=1, column=1, padx=15, pady=15)

        self.method_label = tk.Label(master,
                                     font=("Helvetica", 20, "bold"),
                                     text='search algorithm')
        self.method_label.grid(row=2, column=0, padx=15, pady=15)

        self.method_var = tk.StringVar(master)
        self.method_var.set('bfs')
        self.method_menu = tk.OptionMenu(master, self.method_var, *method_list)
        self.method_menu.config(width=20, font=20)
        self.method_menu.grid(row=2, column=1, padx=15, pady=15)

        self.search_button = tk.Button(master,
                                       text='search path',
                                       command=self.search_path,
                                       font=40,
                                       bg='#0081CF')
        self.search_button.grid(row=3, column=0, columnspan=2, pady=15)

    def search_path(self):
        source_city = self.source_var.get()
        target_city = self.target_var.get()
        algorithm = self.method_var.get()
        heuristic_function = 'L0'
        if algorithm == 'dijkstra':
            algorithm = 'a_star'
            heuristic_function = 'L0'
        elif algorithm == 'a_star(L1)':
            algorithm = 'a_star'
            heuristic_function = 'L1'
        elif algorithm == 'a_star(L2)':
            algorithm = 'a_star'
            heuristic_function = 'L2'
        command = f'run -s {source_city} -t {target_city} -m {algorithm} -h {heuristic_function}'
        result = subprocess.run(command, shell=True, cwd=os.getcwd())
        print(result.stdout)


if __name__ == '__main__':
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()
