import json
from typing import List, Tuple

import geopandas as gpd  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d  # type: ignore
import pyproj
import tqdm  # type: ignore
from shapely.geometry import MultiPolygon, Polygon  # type: ignore


def visualize_geojson(geojson: dict):
    """
    GeoPandasとMatplotlibを用いてGeoJSONを可視化する。

    Args:
        geojson(dict): 可視化するGeoJSON
    """
    gdf = gpd.GeoDataFrame.from_features(geojson["features"])
    gdf.plot()
    plt.show()


def convert_gdf_to_geojson(gdf: gpd.GeoDataFrame) -> dict:
    """
    GeoDataFrameをGeoJSON形式に変換する。

    Args:
        gdf(gpd.GeoDataFrame): geojson形式に変換するGeoDataFrame

    Returns:
        GeoJSON: GeoDataFrameをGeoJSON形式に変換したもの
    """
    return json.loads(gdf.to_json())


def paint_buildings_by_height(gdf: gpd.GeoDataFrame, height_threshold: float = 30):
    """
    GeoDataFrameに含まれる建物の高さに基づいて建物を塗り分ける。

    Args:
        gdf(gpd.GeoDataFrame): 建物のGeoDataFrame
        height_threshold(float): 建物の高さ
    """
    gdf["color"] = "blue"
    gdf.loc[gdf["measuredHeight"] >= height_threshold, "color"] = "red"

    fig, ax = plt.subplots()
    for color in gdf["color"].unique():
        gdf[gdf["color"] == color].plot(ax=ax, color=color)

    plt.show()


def calculate_building_centers(gdf):
    """
    Calculate the centroid of each building from its geometry.

    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame containing building geometries.

    Returns:
        List[tuple]: List of tuples representing the centroids (x, y) of each building.
    """
    centers = []
    for geometry in gdf["geometry"]:
        centroid = geometry.centroid
        centers.append((centroid.x, centroid.y))
    return centers


def serach_center_building(gdf: gpd.GeoDataFrame):
    """
    Calculate the centroid of each building from its geometry.
    """
    latitudes, longitudes = zip(*gdf["building_center"])

    mean_latitude = np.mean(latitudes)
    mean_longitude = np.mean(longitudes)

    distances = np.sqrt(
        (np.array(latitudes) - mean_latitude) ** 2
        + (np.array(longitudes) - mean_longitude) ** 2
    )
    nearest_index = np.argmin(distances)
    mean_distance = np.mean(distances)
    gdf_near = gdf[distances <= mean_distance / 2]

    nearest_building_center = gdf["building_center"].iloc[nearest_index]

    return nearest_index, gdf_near


def paint_center_building(gdf: gpd.GeoDataFrame, nearest_index: int):
    """
    Paint the building at the nearest index to the mean center.
    """
    gdf["color"] = "gray"
    gdf.loc[nearest_index, "color"] = "green"
    fig, ax = plt.subplots()
    for color in gdf["color"].unique():
        gdf[gdf["color"] == color].plot(ax=ax, color=color)
    building_centers = gdf["building_center"].tolist()
    x_coords, y_coords = zip(*building_centers)
    ax.scatter(
        x_coords, y_coords, color="red", s=10
    )  # Plot red dots at building centers
    plt.show()


class CreateCityMesh:
    def __init__(self, building_gdf: gpd.GeoDataFrame):
        self.footprint: Tuple[Polygon, MultiPolygon] = building_gdf["geometry"]
        self.height: float = building_gdf["measuredHeight"]
        self.EPSG4612 = pyproj.Proj("+init=EPSG:4612")
        self.EPSG2451 = pyproj.Proj("+init=EPSG:2451")

    def _create_mesh_triangles(self, num_base_vertices: int) -> List[List[int]]:
        """
        建物の3Dメッシュのtrianglesを作成する。

        Args:
            num_base_vertices(int): 建物の底面の頂点数

        Returns:
            List[List[int]]: 建物の3Dメッシュのtriangles
        """
        # 側面のtrianglesを作成
        triangles = []
        for i in range(num_base_vertices - 1):
            triangles.append([i, (i + 1) % num_base_vertices, i + num_base_vertices])
            triangles.append(
                [
                    (i + 1) % num_base_vertices,
                    (i + 1) % num_base_vertices + num_base_vertices,
                    i + num_base_vertices,
                ]
            )

        # 底面と天井面のtrianglesを作成
        for i in range(2, num_base_vertices):
            # 底面
            triangles.append([0, i - 1, i])
            # 天井面
            triangles.append(
                [num_base_vertices, num_base_vertices + i, num_base_vertices + i - 1]
            )

        return triangles

    def _create_building_mesh(
        self, bldg_footprint: Polygon
    ) -> Tuple[o3d.geometry.TriangleMesh, float]:
        """
        建物のフットプリントの情報から3Dメッシュを作成する。

        Args:
            bldg_footprint(Polygon): 建物のフットプリント

        Returns:
            o3d.geometry.TriangleMesh: 建物の3Dメッシュ
        """
        # フットプリントの外周座標・高さ情報を取得
        exterior_info = np.array(bldg_footprint.exterior.coords)
        exterior = exterior_info[:, :2]

        # 緯度経度から平面直角座標系に変換
        for i in range(len(exterior)):
            x, y = pyproj.transform(
                self.EPSG4612, self.EPSG2451, exterior[i, 0], exterior[i, 1]
            )
            exterior[i] = [x, y]

        # 建物の底面と天井面の頂点を作成
        bottom_vertices = np.hstack([exterior[:-1], np.zeros((len(exterior) - 1, 1))])
        top_vertices = np.hstack(
            [exterior[:-1], np.full((len(exterior) - 1, 1), self.height)]
        )

        # verticesのリストを作成
        vertices = np.vstack([bottom_vertices, top_vertices])
        num_base_vertices = len(bottom_vertices)

        # trianglesを作成
        triangles = self._create_mesh_triangles(num_base_vertices)

        # open3dのTriangleMeshオブジェクトを作成
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()

        return mesh

    def _paint_mesh_by_height(
        self,
        mesh: o3d.geometry.TriangleMesh,
        height_threshold: float = 30,
    ) -> o3d.geometry.TriangleMesh:
        """
        heightがheight_thresholdより大きい場合はマゼンタ、
        height_threshold以下の場合は灰色でメッシュを塗りつぶす。

        Args:
            mesh(o3d.geometry.TriangleMesh): 3Dメッシュ
            height_threshold(float): 3Dメッシュを塗りつぶす高さの閾値

        Returns:
            o3d.geometry.TriangleMesh: 塗りつぶされた3Dメッシュ
        """
        if self.height > height_threshold:
            mesh.paint_uniform_color([1, 0, 1])
        else:
            mesh.paint_uniform_color([0.5, 0.5, 0.5])
        return mesh

    def create_city_mesh(
        self,
        meshes: List[o3d.geometry.TriangleMesh],
    ) -> List[o3d.geometry.TriangleMesh]:
        """
        建物メッシュを作成し、meshes(建物の3Dメッシュの集合)に追加して返す。

        Args:
            meshes (List[o3d.geometry.TriangleMesh]): 建物の3Dメッシュの集合

        Returns:
            List[o3d.geometry.TriangleMesh]: 建物の3Dメッシュの集合
        """
        if isinstance(self.footprint, Polygon):
            mesh = self._create_building_mesh(self.footprint)
            mesh = self._paint_mesh_by_height(mesh, 30)
            meshes.append(mesh)

        elif isinstance(self.footprint, MultiPolygon):
            polygons = [polygon for polygon in self.footprint.geoms]
            mesh_bldg = o3d.geometry.TriangleMesh()
            for i, polygon in enumerate(polygons, start=1):
                mesh = self._create_building_mesh(polygon)
                mesh_bldg += mesh
            mesh_bldg = self._paint_mesh_by_height(mesh_bldg, 30)
            # o3d.visualization.draw_geometries([mesh_bldg], window_name="Building Mesh")
            meshes.append(mesh_bldg)

        return meshes


if __name__ == "__main__":
    gdf = gpd.read_file("data/yokosuka_plateau/udx/bldg/52397533_bldg_6697_op.gml")
    # gdf = gdf.head(10)
    # building_centers = calculate_building_centers(gdf)
    # gdf['building_center'] = building_centers
    # nearest_index, gdf_near = serach_center_building(gdf)
    # gdf = gdf_near
    # paint_center_building(gdf_near, nearest_index)

    print(gdf["storeysAboveGround"])  # 階数
    print(gdf["totalFloorArea"])  # 建物の床面積

    # paint_buildings_by_height(gdf, 30)
    num_buildings = len(gdf.index)
    print(gdf.columns)
    if "measuredHeight" not in gdf.columns:
        print("measuredHeight is not found")
        exit(1)

    print(f"Number of buildings: {num_buildings}")

    meshes: List[o3d.geometry.TriangleMesh] = []
    for i in tqdm.tqdm(range(num_buildings)):
        building_gdf = gdf.iloc[i]
        # meshes = CreateCityMesh(building_gdf).create_city_mesh(meshes)

    o3d.visualization.draw_geometries(meshes, window_name="Building Mesh")
