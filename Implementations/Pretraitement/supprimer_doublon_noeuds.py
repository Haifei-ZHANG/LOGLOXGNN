#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Suppression des doublons stricts
dans un dataset PyTorch Geometric.

Un doublon strict possède :

- mêmes features
- même label
- mêmes voisins

Sortie :

donnees_netoyees/
    Dataset/
        data.pt


Exemple :

python supprimer_doublon.py ^
 --input temp_data/Chameleon/processed/data.pt ^
 --dataset Chameleon
"""


import os
import argparse
import hashlib

import torch
import numpy as np

from collections import defaultdict
from torch_geometric.data import Data



# ==========================================================
# Chargement PyG
# ==========================================================

def load_pyg(path):

    from torch_geometric.data import Data


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


    if isinstance(obj, tuple):

        print("Format InMemoryDataset détecté")


        data_dict = obj[0]


        data = Data()


        for k,v in data_dict.items():

            setattr(
                data,
                k,
                v
            )


        return data


    return obj


# ==========================================================
# Hash
# ==========================================================

def hash_array(x):

    return hashlib.sha1(
        np.ascontiguousarray(x)
        .tobytes()
    ).hexdigest()



# ==========================================================
# Recherche doublons stricts
# ==========================================================

def find_duplicates(data):


    x=data.x.cpu().numpy()

    y=data.y.cpu().numpy()

    edge=data.edge_index.cpu().numpy()


    n=x.shape[0]


    adjacency=[[] for _ in range(n)]


    for u,v in zip(edge[0],edge[1]):

        adjacency[int(u)].append(int(v))



    signatures=defaultdict(list)


    for i in range(n):


        signature=(

            hash_array(x[i]),

            int(y[i]),

            tuple(sorted(adjacency[i]))

        )


        signatures[signature].append(i)



    groups=[

        g for g in signatures.values()

        if len(g)>1

    ]


    return groups



# ==========================================================
# Nettoyage
# ==========================================================

def remove_duplicates(data, groups):


    n=data.x.shape[0]


    keep=np.ones(
        n,
        dtype=bool
    )


    removed=[]


    for g in groups:


        # conserver le premier

        for node in g[1:]:

            keep[node]=False

            removed.append(node)



    mapping={}


    new_id=0


    for old in range(n):


        if keep[old]:

            mapping[old]=new_id

            new_id+=1



    # ==============================
    # nouvelles features
    # ==============================


    data.x=data.x[
        keep
    ]



    data.y=data.y[
        keep
    ]



    # ==============================
    # reconstruction edges
    # ==============================


    edges=data.edge_index.cpu().numpy()


    new_edges=[]


    for u,v in zip(edges[0],edges[1]):


        u=int(u)

        v=int(v)


        if keep[u] and keep[v]:

            new_edges.append(

                [

                    mapping[u],

                    mapping[v]

                ]

            )



    if len(new_edges)>0:


        data.edge_index=torch.tensor(

            np.array(new_edges).T,

            dtype=torch.long

        )


    else:


        data.edge_index=torch.empty(

            (2,0),

            dtype=torch.long

        )



    # ==============================
    # masks
    # ==============================


    for attr in [

        "train_mask",

        "val_mask",

        "test_mask"

    ]:


        if hasattr(data,attr):


            value=getattr(
                data,
                attr
            )


            setattr(

                data,

                attr,

                value[keep]

            )



    return data, removed



# ==========================================================
# Main nettoyage
# ==========================================================

def clean(input_file,dataset):


    print("="*60)

    print("NETTOYAGE :",dataset)

    print("="*60)



    data=load_pyg(input_file)



    print(
        "Avant :",
        data.x.shape[0],
        "noeuds"
    )



    groups=find_duplicates(data)



    print(
        "Groupes doublons :",
        len(groups)
    )



    removed=sum(

        len(g)-1

        for g in groups

    )


    print(
        "Noeuds supprimés :",
        removed
    )



    data,removed_nodes=remove_duplicates(

        data,

        groups

    )



    print(
        "Après :",
        data.x.shape[0],
        "noeuds"
    )



    print(
        "Arêtes :",
        data.edge_index.shape[1]
    )



    output_dir=os.path.join(

        "donnees_netoyees",

        dataset

    )


    os.makedirs(

        output_dir,

        exist_ok=True

    )



    output=os.path.join(

        output_dir,

        "data.pt"

    )



    torch.save(

        data,

        output

    )



    print()

    print(
        "Sauvegardé :",
        output
    )



# ==========================================================
# MAIN
# ==========================================================

def main():


    parser=argparse.ArgumentParser()


    parser.add_argument(

        "--input",

        required=True

    )


    parser.add_argument(

        "--dataset",

        required=True

    )


    args=parser.parse_args()



    clean(

        args.input,

        args.dataset

    )




if __name__=="__main__":

    main()