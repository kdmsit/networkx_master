import networkx as nx
import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

def parse_args():
    parser = argparse.ArgumentParser(description="Networkx Code")

    parser.add_argument('--problem', nargs='?', default='two',help='Give the problem id')
    parser.add_argument('--inputpath', nargs='?', default="../data/lclassico.edgelist", help='Give input file path')
    parser.add_argument('--outputpath', nargs='?', default="../out/spectral_lclassico.png", help='Give the output file path')
    return parser.parse_args()


def problem1(args):
    '''
    Read gml file and save its edgelist as id
    '''
    inputfilePath = args.inputpath
    outputfilepath=args.outputpath
    G = nx.read_gml(inputfilePath)
    nodelist={}
    nodecount=0
    for node in G.nodes:
        nodelist[node]=nodecount
        nodecount=nodecount+1
    edgelist=[]
    for edge in G.edges:
        edgelist.append([nodelist[edge[0]],nodelist[edge[1]]])
    df = pd.DataFrame(edgelist)
    df.to_csv(outputfilepath,sep=" ",header=False,index=False)

def problem2(args):
    '''
    Read edgelist file and do spectral clustering
    '''
    inputfilePath = args.inputpath
    outputfilepath=args.outputpath
    edgelist = pd.read_csv(inputfilePath,sep=" ",header=None).values
    G = nx.Graph()
    G.add_edges_from(edgelist)
    A = nx.laplacian_matrix(G)
    laplacian_Matrix = A.todense()
    eigenValues, eigenVectors = np.linalg.eigh(laplacian_Matrix)
    EigV = eigenVectors.T
    sortedEigenValueIndex = np.argsort(eigenValues)
    secondSmallestEigenVector = EigV[sortedEigenValueIndex[1]]
    kmeans = KMeans(n_clusters=2).fit(np.asanyarray(secondSmallestEigenVector).reshape(-1, 1))
    labels = kmeans.labels_
    print(labels)
    nodeColorMap = []
    for i in range(len(G.nodes)):
        if (labels[i] == 0):
            nodeColorMap.append('blue')
        elif (labels[i] == 1):
            nodeColorMap.append('red')
        elif (labels[i] == 2):
            nodeColorMap.append('green')
        elif (labels[i] == 3):
            nodeColorMap.append('yellow')
    nx.draw(G, node_color=nodeColorMap, with_labels=True)
    plt.savefig(outputfilepath)
    plt.close()
if __name__ == "__main__":
    args = parse_args()
    if args.problem=='one':
        problem1(args)
    elif args.problem=='two':
        problem2(args)
