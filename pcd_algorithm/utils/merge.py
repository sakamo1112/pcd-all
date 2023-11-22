import numpy as np  # type: ignore
import open3d as o3d  # type: ignore


def merge(
    pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud
) -> o3d.geometry.PointCloud:
    """
    2点群pcd1, pcd2を受け取り、それらの点群をマージして返す。

    Args:
        pcd1 (o3d.geometry.PointCloud): 入力点群1
        pcd2 (o3d.geometry.PointCloud): 入力点群2

    Returns:
        o3d.geometry.PointCloud: pcd1とpcd2がマージされた点群
    """
    pcd_merged = o3d.geometry.PointCloud()
    points = np.concatenate([np.array(pcd1.points), np.array(pcd2.points)], 0)
    colors = np.concatenate([np.array(pcd1.colors), np.array(pcd2.colors)], 0)
    pcd_merged.points = o3d.utility.Vector3dVector(points)
    pcd_merged.colors = o3d.utility.Vector3dVector(colors)

    return pcd_merged
