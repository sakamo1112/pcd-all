import argparse
from typing import List

import hdbscan  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d  # type: ignore
from sklearn.cluster import (KMeans, MeanShift,  # type: ignore
                             estimate_bandwidth)


class Clustering:
    def __init__(self):
        pass

    def create_pcd_list(
        self, pcd: o3d.geometry.PointCloud, labels: np.ndarray
    ) -> List[o3d.geometry.PointCloud]:
        """
        点群とラベルのリストを受け取り、クラスタ別に点群を作成後リストにして返す.

        Args:
            pcd (o3d.geometry.PointCloud): 入力点群
            labels (np.ndarray): クラスタラベル

        Returns:
            List[o3d.geometry.PointCloud]: クラスタ点群のリスト
        """
        max_label = labels.max()
        cluster_pcds = []
        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            cluster_pcd = pcd.select_by_index(cluster_indices)
            cluster_pcds.append(cluster_pcd)
        return cluster_pcds

    def paint_clusters(
        self, pcd: o3d.geometry.PointCloud, labels: np.ndarray
    ) -> o3d.geometry.PointCloud:
        """
        クラスタリング結果をもとに点群の色を塗り分ける.

        Args:
            pcd (o3d.geometry.PointCloud): クラスタリング結果を反映する点群
            labels (np.ndarray): クラスタ数

        Returns:
            o3d.geometry.PointCloud: クラスタ別に色を塗った点群
        """
        max_label = labels.max()
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        return pcd

    def draw_BBox(
        self, cluster_pcds: list[o3d.geometry.PointCloud]
    ) -> List[o3d.geometry.OrientedBoundingBox]:
        """
        クラスタに対応するBBoxを描画する.

        Args:
            cluster_pcds (list[o3d.geometry.PointCloud]): クラスタリング結果の点群

        Returns:
            List[o3d.geometry.OrientedBoundingBox]: クラスタリング結果の点群に対応するBBox
        """
        bbox_list = []
        for cluster_pcd in cluster_pcds:
            bbox = cluster_pcd.get_axis_aligned_bounding_box()
            bbox.color = (1, 0, 0)
            bbox_list.append(bbox)
        return bbox_list

    def draw_MeshBox(
        self, cluster_pcds: list[o3d.geometry.PointCloud]
    ) -> List[o3d.geometry.TriangleMesh]:
        """
        クラスタに対応するMeshBoxを描画する.

        Args:
            cluster_pcds (list[o3d.geometry.PointCloud]): クラスタリング結果の点群

        Returns:
            List[o3d.geometry.TriangleMesh]: クラスタリング結果の点群に対応するBBox
        """
        mesh_box_list = []
        for cluster_pcd in cluster_pcds:
            bbox = cluster_pcd.get_axis_aligned_bounding_box()
            extent = bbox.get_extent()
            mesh_box = o3d.geometry.TriangleMesh.create_box(
                width=extent[0], height=extent[1], depth=extent[2]
            )
            mesh_box.paint_uniform_color([0, 1, 0])
            mesh_box.compute_vertex_normals()
            mesh_box.translate(bbox.get_center() - extent / 2)
            mesh_box_list.append(mesh_box)
        return mesh_box_list

    def k_means(
        self, pcd: o3d.geometry.PointCloud, n_clusters: int
    ) -> List[o3d.geometry.PointCloud]:
        """
        受け取った点の集合をk-means法でクラスタリングする.

        Args:
            pcd (o3d.geometry.PointCloud): 入力点群
            num_of_cluster (int): クラスタ数

        Returns:
            List[o3d.geometry.PointCloud]: クラスタリング後の点群リスト
        """
        pcd_points = np.asarray(pcd.points)
        assert len(pcd_points) >= n_clusters, "points is less than number of clusters."
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(pcd_points)

        pcd = self.paint_clusters(pcd, labels)
        cluster_pcds = self.create_pcd_list(pcd, labels)
        bbox_list = self.draw_BBox(cluster_pcds)
        mesh_box_list = self.draw_MeshBox(cluster_pcds)

        o3d.visualization.draw_geometries(cluster_pcds + bbox_list)
        o3d.visualization.draw_geometries(cluster_pcds + mesh_box_list)

        return cluster_pcds

    def mean_shift(
        self, pcd: o3d.geometry.PointCloud, quantile: float, n_samples: int
    ) -> List[o3d.geometry.PointCloud]:
        """
        与えられた点群をMean-Shift法でクラスタリングする。

        Args:
            pcd (o3d.geometry.PointCloud): クラスタリングを行う点群
            quantile (float): バンド幅の推定に使用する分位数
            n_samples (int): バンド幅の推定に使用するサンプル数

        Returns:
            List[o3d.geometry.PointCloud]: クラスタリング後の点群リスト
        """
        pcd_np = np.asarray(pcd.points)
        bandwidth = estimate_bandwidth(pcd_np, quantile=quantile, n_samples=n_samples)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(pcd_np)
        labels = ms.labels_

        pcd = self.paint_clusters(pcd, labels)
        cluster_pcds = self.create_pcd_list(pcd, labels)
        bbox_list = self.draw_BBox(cluster_pcds)
        mesh_box_list = self.draw_MeshBox(cluster_pcds)

        o3d.visualization.draw_geometries(cluster_pcds + bbox_list)
        o3d.visualization.draw_geometries(cluster_pcds + mesh_box_list)

        return cluster_pcds

    def dbscan(
        self, pcd: o3d.geometry.PointCloud, eps: float, min_points: int
    ) -> List[o3d.geometry.PointCloud]:
        """
        DBSCANクラスタリングを実行する関数

        Args:
        pcd (o3d.geometry.PointCloud): クラスタリングを行う点群
        eps (float): コア点判定時に用いる半径の距離
        min_points (int): 半径epsの内側に含む点の最小値

        Returns:
        List[o3d.geometry.PointCloud]: クラスタリング後の点群リスト
        """
        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug
        ) as cm:
            labels = np.array(
                pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True)
            )

        pcd = self.paint_clusters(pcd, labels)
        cluster_pcds = self.create_pcd_list(pcd, labels)
        bbox_list = self.draw_BBox(cluster_pcds)
        mesh_box_list = self.draw_MeshBox(cluster_pcds)

        o3d.visualization.draw_geometries(cluster_pcds + bbox_list)
        o3d.visualization.draw_geometries(cluster_pcds + mesh_box_list)

        return cluster_pcds

    def hdbscan(
        self, pcd: o3d.geometry.PointCloud, min_cluster_size: int
    ) -> List[o3d.geometry.PointCloud]:
        """
        HDBSCANクラスタリングを実行する関数

        Args:
        pcd (o3d.geometry.PointCloud): クラスタリングを行う点群
        min_cluster_size (int): クラスタを構成するための最小点数

        Returns:
        List[o3d.geometry.PointCloud]: クラスタリング後の点群リスト
        """
        pcd_np = np.asarray(pcd.points)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(pcd_np)
        pcd = self.paint_clusters(pcd, labels)
        cluster_pcds = self.create_pcd_list(pcd, labels)
        bbox_list = self.draw_BBox(cluster_pcds)
        mesh_box_list = self.draw_MeshBox(cluster_pcds)

        o3d.visualization.draw_geometries(cluster_pcds + bbox_list)
        o3d.visualization.draw_geometries(cluster_pcds + mesh_box_list)

        return cluster_pcds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and display a point cloud")
    parser.add_argument(
        "--pcd_path",
        type=str,
        default="output.pcd",
        help="Path to the point cloud file (.pcd)",
    )
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.pcd_path)
    # pcd_clustered = Clustering().k_means(pcd, 3)
    # pcd_clustered = Clustering().mean_shift(pcd, 0.05, 40)
    # pcd_clustered = Clustering().dbscan(pcd, 0.5, 50)
    # pcd_clustered = Clustering().hdbscan(pcd, 15)
