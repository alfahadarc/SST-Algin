# SST-Algin
Topological Network Alignment using Self-Supervised Siamese Model


# How to replicate

**Important:** may need to change the path in both `sh` & `py` files according to your file structure

## Add Noise

- Code file name `Noisy_graph.py`
- follow `run_noise.sh` file

*Command Format* `python Noisy_graph.py graph_file_name percentage_error`

## Run ORCA for graphlet
- follow `run_orca.sh` file
- edit the graph file and add `(node_count, edge_count)` at the top of each file before running `orca`  

*Command Format* `orca.exe node [#nodes] filename output`

**Imoprtant:** need to add `orca.exe` to the `bash` path, `orca` software can be found in folder `orca-master` or can be downloaded from github


## Run imap

- edit the graph file and remove `(node_count, edge_count)` at the top of each file before running `imap` code
- Code file name `Graphlet_imap.py`
- follow `run_imap_all.sh` file with other `imap.sh` files

*Command Format* `python Graphlet_imap.py graph_1 graph_2 graphlet_file_1 graphlet_file_2 initial_map nbr`

## Run GATNE
- Code file name `GATNE.py`
- follow `run_gatne_all.sh` file with other `gatne.sh` files

*Command Format* `python GATNE.py g1 g2 f1 f2 mapI mapT lr epc bs dim topk
`