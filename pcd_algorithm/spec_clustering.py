import argparse


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans

class graph_clustering():
    def __init__(self, K, n_clusters, lap_kind):
        """
        K : int
        n_clusters : int
        lap_kind : None or str
            Ex. "rw"
        """
        self.lap_kind = lap_kind
        self.K = K
        self.n_clusters = n_clusters

    def _calc_g_laplacian(self, X):
        D = np.diag(np.sum(X, axis=0))
        L = D - X
        self.L = L
        self.D = D
        return L

    def _calc_rw_g_laplacian(self, X):
        L = self._calc_g_laplacian(X)
        Lrw = np.dot(np.linalg.inv(self.D), L)
        self.L = Lrw
        return Lrw

    def _get_K_eigen_vec(self, Lam, V):
        sorted_index = np.argsort(Lam.real)
        Lam_K = Lam[sorted_index][0:self.K]
        V_K = V[sorted_index][0:self.K]

        self.Lam_K = Lam_K
        self.V_K = V_K

        return Lam_K, V_K

    def _Kmeans_V_K(self, V_K):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(V_K.T.real)
        clusters = kmeans.labels_
        self.clusters = clusters
        return clusters

    def laplacian_clustering(self, X):
        """
        X : ndarray
            a matrix representation of undirected graph
        """
        if self.lap_kind is None:
            L = self._calc_g_laplacian(X)
        elif self.lap_kind == "rw":
            L = self._calc_rw_g_laplacian(X)
        else: 
            raise NotImplementedError

        Lam, V = np.linalg.eig(L)
        Lam_K, V_K = self._get_K_eigen_vec(Lam, V)
        clusters = self._Kmeans_V_K(V_K)
        return clusters

# for plot
def get_labels_dic(G):
    """
    G : networkx.classes.graph.Graph
    """
    labels_dic = { key:G.node[key]['label'] for key in G.node.keys() }
    return labels_dic

def get_labels_list(G):
    labels_list = np.array([ G.node[key]['label'] for key in G.node.keys() ])
    return labels_list

def split_labels(labels_list, clusters):
    """
    labels_list : list
        return value of get_labels_list function.
    clusters : ndarray
        return value of graph_clustering.laplacian_clustering
    """
    class_uniq = np.unique(clusters)
    node_index_split= []
    labels_split = []
    index = np.arange(clusters.shape[0])
    for class_elem in class_uniq:
        node_index_split.append(list(index[clusters == class_elem]))
        labels_split.append(list(labels_list[clusters == class_elem]))
    return node_index_split, labels_split

def plot_clusters(G, node_index_split):
    """
    G : networkx.classes.graph.Graph
    node_index_split : list (2-dim list)
        return value of split_labels function.
    """
    labels_dic = get_labels_dic(G)
    pos = nx.spring_layout(G)
    colors = [ cm.jet(x) for x in np.arange(0, 1, 1/len(node_index_split)) ]
    for i, nodes in enumerate(node_index_split):
        nx.draw_networkx(G, node_color=colors[i], labels=labels_dic, nodelist=nodes, pos=pos, font_color="r")
    plt.show()



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process some point cloud data.")
    parser.add_argument("--pcd_path", type=str, help="Path to the input point cloud data file")
    parser.add_argument("--output_path", type=str, help="Path to save the processed point cloud data")
    args = parser.parse_args()

    data_path = "./data/lesmis/lesmis.gml"
    G = nx.read_gml(data_path)
    X = np.array(nx.to_numpy_matrix(G))

    GClus = graph_clustering(K=15, n_clusters=5, lap_kind="rw")
    clusters = GClus.laplacian_clustering(X)
    labels_list = get_labels_list(G)
    node_index_split, labels_split = split_labels(labels_list, clusters)
    plot_clusters(G, node_index_split)



if __name__ == "__main__":
    main()
