import argparse
from typing import List

import numpy as np
import open3d as o3d  # type: ignore


def create_mesh_edges(cube_size: float, cube_center: np.ndarray) -> o3d.geometry.LineSet:
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


def create_mesh_cube(cube_size: float, cube_center: np.ndarray, cube_color: np.ndarray) -> o3d.geometry.TriangleMesh:
    """
    与えられたサイズと中心点を持つ立方体のメッシュを作成します。(立方体の実体)

    Parameters:
    cube_size (float): 立方体の一辺の長さ
    cube_center (np.ndarray): 立方体の中心点

    Returns:
    o3d.geometry.TriangleMesh: 立方体のメッシュ
    """
    cube = o3d.geometry.TriangleMesh.create_box(width=cube_size, height=cube_size, depth=cube_size)
    cube.translate(cube_center - np.array([cube_size / 2, cube_size / 2, cube_size / 2]))
    cube.vertex_colors = o3d.utility.Vector3dVector(np.tile(cube_color, (8, 1)))
    return cube


def voxel_grid_to_mesh(voxel_grid: o3d.geometry.VoxelGrid, voxel_size: float) -> o3d.geometry.TriangleMesh:
    """
    VoxelGridをTriangleMeshに変換します。

    Parameters:
    voxel_grid (o3d.geometry.VoxelGrid): 入力VoxelGrid
    voxel_size (float): ボクセルの一辺の長さ

    Returns:
    o3d.geometry.TriangleMesh: 結合されたメッシュ
    """
    combined_mesh = o3d.geometry.TriangleMesh()

    # 各ボクセルを立方体メッシュに変換して結合
    for voxel in voxel_grid.get_voxels():
        cube_center = voxel.grid_index * voxel_size + voxel_grid.get_min_bound() + voxel_size / 2
        cube_color = voxel.color
        cube_mesh = create_mesh_cube(voxel_size, cube_center, cube_color)
        combined_mesh += cube_mesh

    return combined_mesh


def remove_voxels(
    points: np.ndarray, colors: np.ndarray, ratio: float
) -> o3d.geometry.VoxelGrid:
    """
    指定された割合でボクセルをランダムに削除し、新しいボクセルグリッドを生成する。

    Parameters:
    points (np.ndarray): 元の点群の座標
    colors (np.ndarray): 元の点群の色
    ratio (float): 削除するボクセルの割合

    Returns:
    o3d.geometry.VoxelGrid: 削除後の新しいボクセルグリッド
    """
    num_voxels_to_remove = int(len(points) * ratio)
    voxels_to_remove_indices = np.random.choice(
        len(points), num_voxels_to_remove, replace=False
    )
    points_to_keep = np.delete(points, voxels_to_remove_indices, axis=0)
    colors_to_keep = np.delete(colors, voxels_to_remove_indices, axis=0)
    colors_to_keep = colors_to_keep / 2

    pcd_new = o3d.geometry.PointCloud()
    pcd_new.points = o3d.utility.Vector3dVector(points_to_keep)
    pcd_new.colors = o3d.utility.Vector3dVector(colors_to_keep)
    new_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd_new, voxel_size=voxel_size
    )

    return new_voxel_grid


def voxelize_pcd(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    if_remove: bool = False,
    remove_ratio: float = 0.9,
) -> tuple[o3d.geometry.VoxelGrid, List[o3d.geometry.LineSet]]:
    """
    点群をボクセル化し、ボクセルグリッドとラインセットのリストを返します。

    Parameters:
    pcd (o3d.geometry.PointCloud): 入力点群
    voxel_size (float): ボクセルの一辺の長さ
    if_remove (bool): True if remove voxels
    ratio (float): remove ratio

    Returns:
    tuple[o3d.geometry.VoxelGrid, List[o3d.geometry.LineSet]]: ボクセルグリッドとラインセットのリスト
    """
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd, voxel_size=voxel_size
    )
    min_bound = voxel_grid.get_min_bound()
    points = np.array(
        [
            (voxel.grid_index * voxel_size + min_bound + voxel_size / 2)
            for voxel in voxel_grid.get_voxels()
        ]
    )
    colors = np.array([voxel.color for voxel in voxel_grid.get_voxels()])
    if if_remove:
        voxel_grid = remove_voxels(points, colors, remove_ratio)

    linesets = []
    for voxel in voxel_grid.get_voxels():
        cube_center = voxel.grid_index * voxel_size + min_bound + voxel_size / 2
        cube = create_mesh_edges(voxel_size, cube_center)
        linesets.append(cube)
    o3d.visualization.draw_geometries([pcd] + linesets + [voxel_grid])

    mesh = voxel_grid_to_mesh(voxel_grid, voxel_size)
    # o3d.io.write_triangle_mesh("data/voxel.ply", mesh)

    return voxel_grid, linesets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and display a point cloud")
    parser.add_argument(
        "--pcd_path",
        "-i",
        type=str,
        default="data/seisenkan_no_slab.pcd",
        help="Path to the point cloud file (.pcd)",
    )
    parser.add_argument(
        "--voxel_size", "-v", type=float, default=0.2, help="voxel size"
    )
    parser.add_argument(
        "--if_remove", "-r", type=bool, default=False, help="True if remove voxels"
    )
    parser.add_argument("--ratio", type=float, default=0.9, help="remove ratio")
    args = parser.parse_args()
    pcd = o3d.io.read_point_cloud(args.pcd_path)
    o3d.visualization.draw_geometries([pcd])
    voxel_size = args.voxel_size

    voxel_grid, linesets = voxelize_pcd(pcd, voxel_size, args.if_remove, args.ratio)
