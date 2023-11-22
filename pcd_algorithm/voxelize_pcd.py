import open3d as o3d
import numpy as np

def voxelize_pcd(pcd, voxel_size):
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    return voxel_grid

if __name__=="__main__":
    pcd = o3d.io.read_point_cloud("output.pcd")
    voxel_grid = voxelize_pcd(pcd, 0.5)
    o3d.visualization.draw_geometries([voxel_grid])
    print("num_of_voxel:", voxel_grid)
    print(np.asarray(voxel_grid.get_voxels()))