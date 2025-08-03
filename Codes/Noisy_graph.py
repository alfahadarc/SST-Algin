import networkx as nx
import random
import sys
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import time
import numpy as np

#Add noise by p
def add_noise(graph, p):
    num_edges = int(len(graph.edges) * p)
    print ("Num of edges to remove:", num_edges)
    non_terminal = [(u, v) for u, v in graph.edges if graph.degree[u] > 1 and graph.degree[v] > 1]
    selected_edges = random.sample(non_terminal, min(num_edges, len(non_terminal)))
    graph.remove_edges_from(selected_edges)
    return graph

#Load graphs filess
def load_graph(graph_file):
    G = nx.Graph();
    for line in graph_file:
        l = line.strip().split();
        G.add_edge(int(l[0]),int(l[1]));
    return G;

#Save noisy graph
def write_graph(graph, filename):
    with open(filename, 'w') as file:
        for u, v in graph.edges:
            file.write(f"{u} {v}\n")

##New noise - add social connection 
# Use BFS to find paths with length exactly h
def find_hop_path(G, nodeA, nodeB, h):
    try:
        paths = list(nx.all_simple_paths(G, nodeA, nodeB, cutoff=h))
        for path in paths:
            if len(path) == h + 1:  # +1 because path includes the starting node
                return True
    except nx.NetworkXNoPath:
        return False
    return False

#Connect two nodes with p probability if they are h-hop connected
def add_connection_noise(G, p, h): ## capture edge count and stop
    print("Adding connection noise")
    for nodeA in G.nodes:
        for nodeB in G.nodes:
            if nodeA != nodeB and not G.has_edge(nodeA, nodeB):  # Check if no direct edge exists
                # Check for a path of length h
                if find_hop_path(G, nodeA, nodeB, h):
                    # Add the edge with probability p
                    if random.random() < p:  # Randomly decide if the edge should be added
                        G.add_edge(nodeA, nodeB)
                        # print(f"{nodeA} and {nodeB} are connected")
    return G

def add_noise_edges(graph, p):
    edges = graph.number_of_edges()
    total_edges = edges + int(edges * p / 100)

    degrees = np.array([graph.degree(node) for node in graph.nodes()])
    sampled_nodes = np.repeat(list(graph.nodes), degrees)

    
    # Continue adding edges until we reach the target number of edges
    while graph.number_of_edges() < total_edges:
        u = random.choice(sampled_nodes)
        adj_u = list(graph.neighbors(u))
        if len(adj_u) <=1:
            continue
        random.shuffle(adj_u)

        if adj_u:  
            v = adj_u[0]  
            adj_v = list(graph.neighbors(v))
            
            if adj_v:
                w = random.choice(adj_v)
                if u != w and not graph.has_edge(u, w):
                    graph.add_edge(u, w)
                    
    return graph

# def show_connection_noise(G,p=0.001,h=2):
#     F=G.copy()
#     nx.draw(F)
#     G = add_connection_noise(G,p,h)
#     nx.draw(G)
#     edit_disttance = nx.graph_edit_distance(G,F)
#     print(f'Edit distance is {edit_disttance}')
#     with PdfPages('graphs.pdf') as pdf:
#         # Plot Graph F
#         plt.figure(figsize=(5, 5))
#         nx.draw(F, with_labels=True, node_color='lightblue', font_weight='bold')
#         plt.title("Graph F")
#         pdf.savefig()  # Saves the current figure to the PDF
#         plt.close()

#         # Plot Graph G
#         plt.figure(figsize=(5, 5))
#         nx.draw(G, with_labels=True, node_color='lightgreen', font_weight='bold')
#         plt.title("Graph G")
#         pdf.savefig()  # Saves the current figure to the PDF
#         plt.close()


            
def main():
    graph_file = open(sys.argv[1])
    p = float(sys.argv[2])
    folder_name = sys.argv[1].split(".txt")[0].split("/")[1]
    file_name = sys.argv[1].split("/")[2].split(".txt")[0]

    logFile="noise_graph.log"
    f = open(logFile, "a")
    f.write("Graph file: " + sys.argv[1] + "\n")
    f.write("Noise percentage: " + str(p) + "\n")
    

    G = load_graph(graph_file)
    ## write nodes and edges number
    f.write("Number of nodes: " + str(G.number_of_nodes()) + "\n")
    f.write("Number of edges: " + str(G.number_of_edges()) + "\n")
    # print("Number of edges bfore adding noise:", G.number_of_edges())
    f.write("Number of edges before adding noise: " + str(G.number_of_edges()) + "\n")

    # print ("Add noise with ", str(p), "%")
    # Noisy_G = add_noise(G,p)
    h=2
    start_time = time.time()
    Noisy_G = add_noise_edges(G,p)
    end_time = time.time()
    # print("Number of edges after adding noise:", G.number_of_edges())
    f.write("Number of edges after adding noise: " + str(G.number_of_edges()) + "\n")
    f.write("Time to add noise: " + str(end_time - start_time) + " seconds\n")

    fn = "30_more/data/" + folder_name + "/" + file_name + "_" + str(p) + ".txt"
    f.write("**Noisy graph saved to: " + fn + "**\n\n\n")
    write_graph(Noisy_G,fn)
    # print (fn)

    #new test
    
    
       
main();
