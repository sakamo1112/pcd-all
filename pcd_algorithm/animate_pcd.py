import numpy as np
import open3d as o3d  # type: ignore


# https://qiita.com/popondeli/items/c20fa0af1ab1f4038f0d
# ここを参考にする
def animate():
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()

    for i in range(1000):
        # アニメーションの各フレームで点群データを更新
        points = np.random.rand(100, 3)
        pcd.points = o3d.utility.Vector3dVector(points)
        vis.add_geometry(pcd)

        # ビジュアライザに点群データを送り、画面に表示
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # キャプチャした画像を保存
        # vis.capture_screen_image("result/frame_{:05d}.png".format(i))

    vis.destroy_window()


animate()
