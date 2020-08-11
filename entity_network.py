#!/usr/bin/env python

import networkx as nx
import matplotlib.pyplot as plt
import os

def draw_graph(pairs,filename):
    k_graph = nx.from_pandas_edgelist(pairs, 'Subject', 'Object', create_using=nx.MultiDiGraph())
    node_deg = nx.degree(k_graph)
    layout = nx.spring_layout(k_graph,k=0.15,iterations=20)

    plt.figure(num=None, figsize=(200, 150),dpi=100)
    nx.draw_networkx(
            k_graph,
            node_size=[int(deg[1]) * 1000 for deg in node_deg],
            arrowsize=20,
            linewidths=1.5,
            pos=layout,
            edge_color='red',
            edgecolors='blue',
            node_color='white',
            font_weight='bold',
            font_size=40
            )
    labels = dict(zip(list(zip(pairs.Subject, pairs.Object)),pairs['Relation'].tolist()))
    nx.draw_networkx_edge_labels(k_graph, pos=layout, edge_labels=labels,font_color='red',font_size=35)
    plt.axis('off')
    plt.savefig(os.path.join("NER","pdf_json_links",filename[0:-4]+"pdf"))



def filter_graph(pairs,node):
    G = nx.from_pandas_edgelist(pairs, 'Subject', 'Object', create_using=nx.MultiDiGraph())
    edges = nx.dfs_successors(G, node)
    nodes = []
    for k, v in edges.items():
        nodes.extend([k])
        nodes.extend(v)
    subgraph = G.subgraph(nodes)
    layout = nx.spring_layout(G,iterations=30)
    plt.figure(num=None, figsize=(48, 36),dpi=100)
    node_deg = nx.degree(G)
    nx.draw_networkx(
            subgraph,
            node_size=300,
            arrowsize=10,
            linewidths=1.5,
            pos=layout,
            edge_color='red',
            edgecolors='blue',
            node_color='white',
            font_weight='bold',
            font_size=14
            )
    labels = dict(zip((list(zip(pairs.Subject, pairs.Object))), pairs['Relation'].tolist()))
    edges= tuple(subgraph.out_edges(data=False))
    sublabels ={k: labels[k] for k in edges}
    nx.draw_networkx_edge_labels(subgraph, pos=layout, edge_labels=sublabels, font_color='red',font_size=14)
    plt.axis('off')
    plt.savefig("Query.pdf")
