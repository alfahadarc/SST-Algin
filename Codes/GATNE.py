import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import random
import os
from torch.utils.data import Dataset
from scipy.spatial import KDTree
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np
import sys
import random 
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ## NOTE
# read graph file as edge-list format
def read_edges(file_path):
    edge_index = [[], []]
    with open(file_path, 'r') as file:
        for line in file:
            u, v = map(int, line.strip().split(" "))
            edge_index[0].append(u)
            edge_index[1].append(v)
    return torch.tensor(edge_index, dtype=torch.long)

# read GC features from Ocra output
def read_features(file_path):
    features = []
    with open(file_path, 'r') as file:
        for line in file:
            features.append(list(map(float, line.strip().split(" "))))
    return torch.tensor(features, dtype=torch.float)

# read acutal_mapping from GC+HM output
def read_map(file_name):
    graph_map = pd.read_csv(file_name, header=None, sep=" ")
    return {int(row[0]): int(row[1]) for idx, row in graph_map.iterrows()}

# dataset constructor: g1_data, g1_data, mapping, and lables 
class GDataset(Dataset):
    def __init__(self, g1_data, g2_data, mapping, labels):
        self.g1_data = g1_data
        self.g2_data = g2_data
        self.mapping = mapping
        self.labels = labels

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        i, j = self.mapping[idx]
        label = self.labels[idx]
        return i, j, label

    
# read init_map from GC+HM output
# generate neg_samples and lables from init_map
def read_init_map_with_labels(file_path, num_nodes_g1, num_nodes_g2, n):
    pos_mapping = []
    # with open(file_path, 'r') as file:
    #     for line in file:
    #         i, j = map(int, line.strip().split())
    df = pd.read_csv(file_path, header=None)  # assuming no header in the file
    
    # Iterate through the rows of the DataFrame and extract the values
    for index, row in df.iterrows():
        i, j = row[0], row[1]  # assuming columns 0 and 1 contain the i, j values
        pos_mapping.append((int(i), int(j)))
    neg_mapping = generate_neg_samples(pos_mapping, num_nodes_g1, num_nodes_g2, n)
    mapping = pos_mapping + neg_mapping
    labels = generate_labels(mapping, pos_mapping)
    return mapping, labels

# generate neg_samples 
def generate_neg_samples(pos_mapping, num_nodes_g1, num_nodes_g2, n):
    neg_mapping = []
    for (i, j) in pos_mapping:
        for _ in range(n):
            while True:
                k = random.randint(0, num_nodes_g2 - 1)
                if k != j: 
                    neg_mapping.append((i, k))
                    break
    return neg_mapping

# assing 0/1 lables for pos_map/neg_map
def generate_labels(mapping, pos_mapping):
    labels = []
    for pair in mapping:
        if pair in pos_mapping:
            labels.append(1)
        else:
            labels.append(0)
    return labels

# set model architecture
class SiameseGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SiameseGNN, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        return x

    def encode(self, g1_data, g2_data):
        g1_embed = self.forward(g1_data)
        g2_embed = self.forward(g2_data)
        return g1_embed, g2_embed

# loss function: contrastive_loss
def contrastive_loss(g1_embed, g2_embed, mapping, labels, margin=1.0):
    loss = 0
    for (i, j), label in zip(mapping, labels):
        diff = g1_embed[i] - g2_embed[j]
        dist = torch.norm(diff, p=2)
        if label == 1:
            loss += dist ** 2
        else:
            loss += torch.relu(margin - dist) ** 2
    return loss / len(mapping)

# train model
def train(model, train_loader, optimizer, g1_data, g2_data):
    model.train()
    total_loss = 0
    for batch in train_loader:
        i, j, labels = batch
        i, j, labels = i.to(device), j.to(device), labels.to(device) ## NOTE
        optimizer.zero_grad()
        g1_embed, g2_embed = model.encode(g1_data, g2_data)
        loss = contrastive_loss(g1_embed, g2_embed, list(zip(i, j)), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# test model
def test(model, test_loader, optimizer, g1_data, g2_data):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            i, j, labels = batch
            g1_embed, g2_embed = model.encode(g1_data, g2_data)
            loss = contrastive_loss(g1_embed, g2_embed, list(zip(i, j)), labels)
            total_loss += loss.item()
    return total_loss / len(test_loader)


def main():
    # read data
    print ("----------------------------------------------------------")
    g1_edges = read_edges(sys.argv[1])
    print ("g1_edges:", g1_edges.shape)
    g1_features = read_features(sys.argv[3])
    print ("g1_features:", g1_features.shape)
    fn1 = sys.argv[1].split("/")[2].split(".txt")[0]

    g2_edges = read_edges( sys.argv[2])
    print ("g2_edges:", g2_edges.shape)
    g2_features = read_features( sys.argv[4])
    print ("g2_features:", g2_features.shape)
    fn2 = sys.argv[2].split("/")[2].split(".txt")[0]

    init_map = sys.argv[5]
    true_mappings = read_map(sys.argv[6])
    print ("true_mappings", len(true_mappings))
    print (fn1)
    print (fn2)
    print ("----------------------------------------------------------")

    # generate neg_samples with lables
    mapping, labels = read_init_map_with_labels(init_map,g1_features.shape[0],g1_features.shape[0],10)

    g1_data = Data(x=g1_features.to(device), edge_index=g1_edges.to(device)) ## NOTE
    g2_data = Data(x=g2_features.to(device), edge_index=g2_edges.to(device))

    # shuffle and split
    indices = list(range(len(mapping)))
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_indices = indices[:split]
    test_indices = indices[split:]

    train_mapping = [mapping[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_mapping = [mapping[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    train_dataset = GDataset(g1_data, g2_data, train_mapping, train_labels)
    test_dataset = GDataset(g1_data, g2_data, test_mapping, test_labels)

    # read model parm.
    lr = float(sys.argv[7])
    epc = int(sys.argv[8])
    bs = int(sys.argv[9])
    input_dim = g1_features.shape[1]
    hidden_dim = 64
    output_dim = int(sys.argv[10])

    print ("lr:", lr, "|epochs:", epc, "|batchzise:", bs)
    print ("input_dim:", input_dim, "|output_dim:", output_dim)

    # set model

    model = SiameseGNN(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr= lr)
    print ("----------------------------------------------------------")

    # start training
    print ("Training..")
    train_loss_lst = []
    test_loss_lst = []
    start_time = time.time()
    for epoch in range(epc):
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
        train_loss = train(model, train_loader, optimizer, g1_data, g2_data)
        train_loss_lst.append(train_loss)
        test_loss = test(model, test_loader, optimizer, g1_data, g2_data)
        test_loss_lst.append(test_loss)
        print(f'Epoch {epoch}, Train Loss: {round(train_loss,5)}, Test Loss: {round(test_loss,5)}')

    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Training completed in {training_duration} seconds.")

    # save plot
    fn = fn1 + "_" + fn2 + "lr_" + str(lr) + "_epc_" + str(epc) +".png"
    epochs_lst = range(1, len(train_loss_lst) + 1)
    plt.plot(epochs_lst, train_loss_lst, 'b', label='Training loss')
    plt.plot(epochs_lst, test_loss_lst, 'r', label='Test loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig("output/"+ fn)
    print ("----------------------------------------------------------")

    # save embedding 
    emb1, emb2= model.encode(g1_data, g2_data)
    emb1 = emb1.cpu().detach().numpy()
    emb2 = emb2.cpu().detach().numpy()

    emb_fn1 = "embd/" + fn1 + "_lr_" + str(lr) + "_epc_" + str(epc) + ".csv"
    emb_fn2 = "embd/" + fn2 + "_lr_" + str(lr) + "_epc_" + str(epc) + ".csv"

    np.savetxt(emb_fn1, emb1, delimiter=',')
    np.savetxt(emb_fn2, emb2, delimiter=',')

    # report topl and top3 acc.
    print("Matching..")
    true_mappings = true_mappings.items()
    kdtree_embd2 = KDTree(emb2)
    _, nn_indices_k1 = kdtree_embd2.query(emb1, k=1)
    _, nn_indices_k3 = kdtree_embd2.query(emb1, k=3)
    _, nn_indices_k5 = kdtree_embd2.query(emb1, k=5)
    _, nn_indices_k10 = kdtree_embd2.query(emb1, k=10)
    
    actual_embd2_indices = np.array([mapping[1] for mapping in true_mappings])
    accuracy_k1 = np.mean(nn_indices_k1.flatten() == actual_embd2_indices)
    print("Accuracy (k=1):", round(accuracy_k1,8))

    nn_indices_k3_list = nn_indices_k3.tolist()
    accuracy_k3 = np.mean([mapping[1] in nn_indices_k3_list[mapping[0]] for mapping in true_mappings])
    print("Accuracy (k=3):", round(accuracy_k3,8))

    nn_indices_k5_list = nn_indices_k5.tolist()
    accuracy_k5 = np.mean([mapping[1] in nn_indices_k5_list[mapping[0]] for mapping in true_mappings])
    print("Accuracy (k=5):", round(accuracy_k5,8))

    nn_indices_k10_list = nn_indices_k10.tolist()
    accuracy_k10 = np.mean([mapping[1] in nn_indices_k10_list[mapping[0]] for mapping in true_mappings])
    print("Accuracy (k=10):", round(accuracy_k10,8))

main();
