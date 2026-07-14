#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert PyTorch Geometric data.pt
to Hamilton GraphSAGE format

Compatible:
- Cora
- CiteSeer
- PubMed
- Actor
- Chameleon
- Squirrel
- WebKB
- datasets with multiple train/val/test splits

Output:
    dataset-G.json
    dataset-id_map.json
    dataset-class_map.json
    dataset-feats.npy
"""


import os
import json
import argparse

import torch
import numpy as np
import networkx as nx



# ============================================================
# Load PyG data.pt
# ============================================================

def load_pyg_data(path):

    print("Loading:", path)


    try:

        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

    except TypeError:

        obj = torch.load(
            path,
            map_location="cpu"
        )


    print("Loaded object:", type(obj))


    # PyG InMemoryDataset
    if isinstance(obj, tuple):

        print("Detected PyG tuple format")


        data_dict = obj[0]


        class DataContainer:
            pass


        data = DataContainer()


        for key,value in data_dict.items():

            setattr(
                data,
                key,
                value
            )


    else:

        print("Detected PyG Data object")

        data=obj


    return data



# ============================================================
# Tensor -> numpy
# ============================================================

def tensor_numpy(x):

    if torch.is_tensor(x):

        return x.detach().cpu().numpy()

    return np.asarray(x)



# ============================================================
# Gestion des masques
# ============================================================

def extract_mask(mask, index, split_id=0):

    """
    Compatible:

    Cas classique:
        mask.shape = [N]

    Cas multi split:
        mask.shape = [N,S]

    """

    value = mask[index]


    if torch.is_tensor(value):

        # plusieurs splits
        if value.dim() > 0 and value.numel() > 1:

            value = value[split_id]


        return bool(value.item())


    return bool(value)



# ============================================================
# Build NetworkX graph
# ============================================================

def build_graph(data, split_id=0):


    print("Building graph...")


    G = nx.Graph()


    num_nodes=data.x.shape[0]


    print("Nodes:",num_nodes)



    for i in range(num_nodes):


        val=False
        test=False


        if hasattr(data,"val_mask"):

            val = extract_mask(
                data.val_mask,
                i,
                split_id
            )


        if hasattr(data,"test_mask"):

            test = extract_mask(
                data.test_mask,
                i,
                split_id
            )



        G.add_node(

            i,

            id=int(i),

            val=val,

            test=test

        )



    edges=tensor_numpy(
        data.edge_index
    )


    for src,dst in zip(edges[0],edges[1]):


        G.add_edge(

            int(src),

            int(dst)

        )


    print(
        "Edges:",
        G.number_of_edges()
    )


    return G



# ============================================================
# Save GraphSAGE Hamilton JSON
# ============================================================

def save_graph_json(G,path):


    nodes=[]


    for node,attrs in G.nodes(data=True):


        nodes.append(

            {

                "id":int(node),

                "val":bool(attrs.get("val",False)),

                "test":bool(attrs.get("test",False))

            }

        )



    links=[]


    for src,dst in G.edges():

        links.append(

            {

                "source":int(src),

                "target":int(dst)

            }

        )



    graph={

        "directed":False,

        "graph":{},

        "nodes":nodes,

        "links":links

    }



    with open(path,"w") as f:

        json.dump(
            graph,
            f
        )



# ============================================================
# Save id map
# ============================================================

def save_id_map(num_nodes,path):


    mapping={}


    for i in range(num_nodes):

        mapping[str(i)] = i



    with open(path,"w") as f:

        json.dump(
            mapping,
            f
        )



# ============================================================
# Save features
# ============================================================

def save_features(data,path):


    np.save(

        path,

        tensor_numpy(data.x)

    )



# ============================================================
# Save labels
# ============================================================

def save_classes(data,path):


    labels=tensor_numpy(
        data.y
    )


    class_map={}


    for i,label in enumerate(labels):


        class_map[str(i)] = int(label)



    with open(path,"w") as f:

        json.dump(
            class_map,
            f
        )



# ============================================================
# Main conversion
# ============================================================

def convert(input_file, output, name, split_id):


    print()
    print("==============================")
    print("Dataset:",name)
    print("==============================")


    data=load_pyg_data(
        input_file
    )


    print()

    print(
        "Features:",
        tuple(data.x.shape)
    )


    print(
        "Labels:",
        tuple(data.y.shape)
    )



    G=build_graph(
        data,
        split_id
    )


    os.makedirs(
        output,
        exist_ok=True
    )


    prefix=os.path.join(
        output,
        name
    )



    print("Saving graph...")


    save_graph_json(

        G,

        prefix+"-G.json"

    )



    print("Saving id map...")


    save_id_map(

        data.x.shape[0],

        prefix+"-id_map.json"

    )



    print("Saving features...")


    save_features(

        data,

        prefix+"-feats.npy"

    )



    print("Saving classes...")


    save_classes(

        data,

        prefix+"-class_map.json"

    )



    print()

    print("==============================")

    print("DONE:",name)

    print("==============================")



# ============================================================
# CLI
# ============================================================

def main():


    parser=argparse.ArgumentParser()



    parser.add_argument(

        "--input",

        required=True,

        help="PyG data.pt path"

    )



    parser.add_argument(

        "--output",

        required=True,

        help="GraphSAGE output directory"

    )



    parser.add_argument(

        "--name",

        required=True,

        help="dataset prefix"

    )



    parser.add_argument(

        "--split_id",

        type=int,

        default=0,

        help="split used for multi-split datasets"

    )



    args=parser.parse_args()



    convert(

        args.input,

        args.output,

        args.name,

        args.split_id

    )



if __name__=="__main__":

    main()