import argparse

import cv2  # type: ignore
import numpy as np
import open3d as o3d  # type: ignore
from voxelize_pcd import voxelize_pcd  # type: ignore


def create_map(voxel_grid, grid_index_array):
    img_size = np.max(grid_index_array, axis=0) - np.min(grid_index_array, axis=0) + 1
    image = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    for voxel in voxel_grid.get_voxels():
        grid_index = voxel.grid_index
        grid_color = np.array([voxel.color[2], voxel.color[1], voxel.color[0]]) * 255
        if (args.dam_mode == True) and (grid_index[2] < args.flood_height):
            image[grid_index[0], grid_index[1]] = [255, 20, 0]
        else:
            image[grid_index[0], grid_index[1]] = grid_color

    cv2.imwrite("image.png", image)


def voxel2pcd(voxel_grid):
    min_bound = voxel_grid.get_min_bound()
    points = []
    colors = []
    points_sea = []
    colors_sea = []
    grid_index_list = []
    for voxel in voxel_grid.get_voxels():
        grid_index_list.append(np.array(voxel.grid_index))
        point = voxel.grid_index * args.voxel_size + min_bound
        if (args.dam_mode == True) and (point[2] < args.flood_height):
            points_sea.append([point[0], point[1], args.flood_height])
            colors_sea.append([0, 20, 255])
        points.append(voxel.grid_index * args.voxel_size + min_bound)
        colors.append(voxel.color)

    pcd_voxel = o3d.geometry.PointCloud()
    pcd_voxel.points = o3d.utility.Vector3dVector(points)
    pcd_voxel.colors = o3d.utility.Vector3dVector(colors)
    if points_sea != []:
        pcd_sea = o3d.geometry.PointCloud()
        pcd_sea.points = o3d.utility.Vector3dVector(points_sea)
        pcd_sea.colors = o3d.utility.Vector3dVector(colors_sea)
        o3d.visualization.draw_geometries([pcd_voxel, pcd_sea])

    return pcd_voxel, np.array(grid_index_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pcd_path",
        type=str,
        default="data/sasebo_kiritori.pcd",
        help="Path to the point cloud file (.pcd)",
    )
    parser.add_argument("--voxel_size", type=float, default=10)
    parser.add_argument("--dam_mode", type=bool, default=False)
    parser.add_argument("--flood_height", type=float, default=90)
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.pcd_path)

    voxel_grid, linesets = voxelize_pcd(pcd, args.voxel_size)

    pcd_voxel, grid_index_array = voxel2pcd(voxel_grid)
    create_map(voxel_grid, grid_index_array)
