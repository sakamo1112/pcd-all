import sys
from os.path import abspath, dirname

import numpy as np
import open3d as o3d

parent_dir = dirname(dirname(dirname(abspath(__file__))))  # type: ignore
if parent_dir not in sys.path:  # type: ignore
    sys.path.append(parent_dir)
print(parent_dir)

from pcd_algorithm.utils.merge import merge


def visualize_cluster(cluster_list: list[list[float]], num_of_cluster: int):
    """
    クラスタリング結果を表示する.

    Args:
        cluster_list (list[list[float]]): 入力点群
        num_of_cluster (int): クラスタ数

    """
    cluster_pcd = o3d.geometry.PointCloud()
    for i in range(num_of_cluster):
        cluster = o3d.geometry.PointCloud()
        cluster.points = o3d.utility.Vector3dVector(cluster_list[i])
        cluster.paint_uniform_color(np.random.rand(3))
        cluster_pcd = merge(cluster_pcd, cluster)

    o3d.visualization.draw_geometries([cluster_pcd])


def k_means(points: np.ndarray, num_of_cluster: int):
    """
    受け取った点の集合をk-means法でクラスタリングする.

    Args:
        points (np.ndarray): 入力点群
        num_of_cluster (int): クラスタ数

    """
    assert len(points) >= num_of_cluster, "points is less than number of clusters."

    seed_points = np.random.rand(num_of_cluster, 3)  # ランダムに点選択するよう変更
    seed_dist = np.array([10e7, 10e7, 10e7])

    while np.max(seed_dist) > 0.005:
        cluster_list: list[list[float]] = [[]] * num_of_cluster

        # 各点をクラスタに分ける
        for point in points:
            distance = []
            for seed_point in seed_points:
                distance = np.append(distance, np.linalg.norm(point - seed_point))
            nearest_index = np.argmin(distance)
            if len(cluster_list[nearest_index]) > 0:
                cluster_list[nearest_index] = np.vstack(
                    [cluster_list[nearest_index], point]
                )
            else:
                cluster_list[nearest_index] = point

        # seed点の更新
        cluster_heart_point = [[0, 0, 0]] * num_of_cluster
        for i in range(num_of_cluster):
            sum_points = np.sum(cluster_list[i], axis=0)
            cluster_heart_point[i] = [
                point / len(cluster_list[i]) for point in sum_points
            ]
            seed_dist = np.linalg.norm(cluster_heart_point[i] - seed_points[i])
        seed_points = np.array(cluster_heart_point)

    visualize_cluster(cluster_list, num_of_cluster)


if __name__ == "__main__":
    points = np.random.rand(100, 3)
    k_means(points, 4)
