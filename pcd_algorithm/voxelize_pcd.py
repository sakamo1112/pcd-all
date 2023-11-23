import argparse
from typing import List

import numpy as np
import open3d as o3d  # type: ignore


def create_mesh_cube(cube_size: float, cube_center: np.ndarray) -> o3d.geometry.LineSet:
    """
    与えられたサイズと中心点を持つ立方体のメッシュを作成します。(メッシュの枠線)

    Parameters:
    cube_size (float): 立方体の一辺の長さ
    cube_center (np.ndarray): 立方体の中心点

    Returns:
    o3d.geometry.LineSet: 立方体のメッシュ
    """
    vertices = (
        np.array(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ]
        )
        * cube_size
        / 2
        + cube_center
    )

    edges = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [1, 3],
            [1, 5],
            [2, 3],
            [2, 6],
            [3, 7],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
        ]
    )

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(vertices)
    lines.lines = o3d.utility.Vector2iVector(edges)
    return lines


def voxelize_pcd(
    pcd: o3d.geometry.PointCloud, voxel_size: float
) -> tuple[o3d.geometry.VoxelGrid, List[o3d.geometry.LineSet]]:
    """
    点群をボクセル化し、ボクセルグリッドとラインセットのリストを返します。

    Parameters:
    pcd (o3d.geometry.PointCloud): 入力点群
    voxel_size (float): ボクセルの一辺の長さ

    Returns:
    tuple[o3d.geometry.VoxelGrid, List[o3d.geometry.LineSet]]: ボクセルグリッドとラインセットのリスト
    """
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd, voxel_size=voxel_size
    )
    min_bound = voxel_grid.get_min_bound()

    linesets = []
    for voxel in voxel_grid.get_voxels():
        cube_center = voxel.grid_index * voxel_size + min_bound + voxel_size / 2
        cube = create_mesh_cube(voxel_size, cube_center)
        linesets.append(cube)
    o3d.visualization.draw_geometries(linesets + [voxel_grid])

    return voxel_grid, linesets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and display a point cloud")
    parser.add_argument(
        "--pcd_path",
        type=str,
        default="data/seisenkan_no_slab.pcd",
        help="Path to the point cloud file (.pcd)",
    )
    parser.add_argument("--voxel_size", type=float, default=0.1, help="voxel size")
    args = parser.parse_args()
    pcd = o3d.io.read_point_cloud(args.pcd_path)
    voxel_size = args.voxel_size

    voxel_grid, linesets = voxelize_pcd(pcd, voxel_size)
