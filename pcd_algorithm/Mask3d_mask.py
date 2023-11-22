import argparse

import numpy as np
import open3d as o3d

SCANNET_COLOR_MAP_20 = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}


def project_Mask3D_to_pcd(pcd_path: str, Prefix: str, result_file: str):
    """
    Mask3Dで得られた結果のテキストファイルの中身を点群データに反映させる。

    Args:
        pcd_path (str): Mask3Dを適用した点群データ
        Prefix (str): 結果が入ったファイルの格納場所
        result_file (str): 結果ファイルの名前(pathではない)

    Returns
        None
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    o3d.visualization.draw_geometries([pcd])

    colors = np.array(pcd.colors)
    with open(Prefix + result_file) as result_f:
        for line in result_f:
            (filename, label_num, prob) = line.split(" ")
            obj_color = np.array(SCANNET_COLOR_MAP_20[int(label_num)]) / 255

            with open(Prefix + filename) as obj_f:
                for i, is_object in enumerate(obj_f):
                    if int(is_object):
                        colors[i] = obj_color

    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud("output.pcd", pcd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd_path", type=str, default="data/Mask3d_result/room.ply")
    parser.add_argument("--Prefix", type=str, default="data/Mask3D_result/")
    parser.add_argument(
        "--result_file_name", type=str, default="20231108_120945_room.txt"
    )
    args = parser.parse_args()

    project_Mask3D_to_pcd(args.pcd_path, args.Prefix, args.result_file_name)
