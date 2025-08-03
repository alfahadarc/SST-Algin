'''
Author: Aljohara Almulhim
IUPUI 2024
'''

import sys;
import networkx as nx;
from numpy import linalg as LA
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
import math;
import datetime;
from sklearn.neighbors import KDTree
from scipy import sparse
import csv
from operator import itemgetter
from lapsolver import solve_dense
from collections import Counter
import pandas as pd

#Load graphs filess
def load_graph(file2read):
    G = nx.Graph();
    for line in file2read:
        l = line.strip().split();
        G.add_edge(int(l[0]),int(l[1]));
    return G;

#Load graphlets files
def load_graphlet(filetoread):
    data = [];
    for line in filetoread:
        line = line.strip().split();
        d = list(map(float,line))
        log_d= [np.log(i) if i!=0.0 else (i) for i in d]
        data.append(log_d);                        
    return np.array(data);

#True Assignment (Mapping File)
def load_map(f):
    lines=f.readlines()
    map_d=[]
    for x in lines:
        map_d.append(int(x.split(' ')[1]))
    f.close()
    return np.array(map_d, dtype=np.int32)

#Process graphlets vectors
#graphlet_vector(u) = avg(graphlet_vectors(nbr_with_2hops)

def compute_2hop_nbrs(G):
    hop2_nbrs = {}
    for u in G.nodes():
        nbrs = set(G.neighbors(u))
        hop2 = set(nbrs)
        for u_i in nbrs:
            hop2.update(G.neighbors(u_i))
        hop2.discard(u)
        hop2_nbrs[u] = list(hop2)
    return hop2_nbrs

def avg_graphlet_vectors_2hops(G, data):
    data = np.asarray(data)
    data_new = np.zeros_like(data)
    hop2_nbrs = compute_2hop_nbrs(G)
    for u in G.nodes():
        nbrs = hop2_nbrs[u]
        if nbrs:
            nbrs_features = data[nbrs].sum(axis=0)
            data_new[u] = nbrs_features / len(nbrs)
        else:
            data_new[u] = data[u]
    return data_new

#Calculate L2 Norm for cost matrix 1 - with kd tree
def kd_tree_map(N1,N2,g1_features,g2_features,top_n):
        cost_m= np.full((N1, N2), np.nan)
        tree = KDTree(g2_features) ## probabilistic count TODO
        dist, ind = tree.query(g1_features, k=top_n) # is true map node here?
        for i in range(N1):
                for nbr in range(top_n):
                        j=ind[i,nbr]
                        dst= dist[i,nbr]
                        cost_m[i,j]=dst
        return cost_m

def main():
        if len(sys.argv) < 7:
                print ("enter graph_file1 graph_file2 graphlet_file1 graphlet_file2 map_file new_map_file nbr");
                sys.exit(0);

        graph_file1 = open(sys.argv[1]);
        graph_file2 = open(sys.argv[2]);
        graphlet_file1 = open(sys.argv[3]);
        graphlet_file2 = open(sys.argv[4]);
        map_file= open(sys.argv[5]);
        top_n= int(sys.argv[6]);

        name1 = sys.argv[1].strip().rsplit(".",1)[0].strip().rsplit("/",1)[-1];
        name2 = sys.argv[2].strip().rsplit(".",1)[0].strip().rsplit("/",1)[-1];

        nn = "sparse_cost_"+name1+'_'+name2+'_top_n_'
        
        name_p = sys.argv[1].split("/")[1].split("/")[0]
        print ("name_path_folder:", name_p)

        #Read Parms.
        print ("---------------------------------------------")
        print ("Path_folder:", name_p)
        print ("dataset:", nn)

        G1 = load_graph(graph_file1);
        G2 = load_graph(graph_file2);
        data1 = load_graphlet(graphlet_file1);
        data2 = load_graphlet(graphlet_file2);
        map_data = load_map(map_file);
        print (map_data.shape[0])
        # map_data_new = load_new_map(new_map_file) ## Where is this function?
        # print (len(map_data_new))
        
        data1_new = avg_graphlet_vectors_2hops(G1, data1)
        print (data1_new.shape)
        data2_new = avg_graphlet_vectors_2hops(G2, data2)
        print (data2_new.shape)
    
        N1= data1.shape[0];
        N2= data2.shape[0];
        sys.stdout.flush();
        print ("---------------------------------------------")

        #Start Calculating ..

        #C1: calculate L2-norm cost using kd-tree
        start_t = datetime.datetime.now();
        cost_m1= kd_tree_map(N1,N2,data1_new,data2_new,top_n)
        #print (cost_m1)
        #cost_m1= kd_tree_map(N1,N2,data1,data2,top_n)
        end_t = datetime.datetime.now();
        print ("time for distance based cost cal. using kd-tree: "+str((end_t - start_t).total_seconds()));

        #M1: mapping using lapsolver
        print ("mapping1 - using lapsolver:")
        row_ind=[]
        col_ind=[]
        start_t = datetime.datetime.now();
        row_ind, col_ind=  solve_dense(cost_m1) ## TODO stop at 80% ##effect of noise at graphlet count
        end_t = datetime.datetime.now();
        print ("time for mapping1= "+ str((end_t - start_t).total_seconds()))
        print ("cost1=",cost_m1[row_ind, col_ind].sum())
        print ("Number of mapped items:",len(row_ind))

        print("map data shape:", map_data.shape)
        print("col_ind shape:", col_ind.shape)
        
        acc1= accuracy_score(map_data, col_ind)
        print ("Accuracy1:", acc1)

        # get initial corrected mapping (as landmark_nodes)
        landmark_map = {'g1_indx': row_ind,'g2_indx': col_ind}
        # print("landmark_map:", landmark_map)
        df = pd.DataFrame(landmark_map)
        print ("initial_mapping:")
        print (df.head(10))

        fn = "init_map/" + name_p + "/init_map_" + name1 + "_" + name2 + ".csv" 
        df.to_csv(fn, index=False, header=False)

main();

