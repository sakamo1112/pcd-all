# 🌈 pcd_all 🌈
点群を扱う処理であれば何でも上げるリポジトリです.
<br><br>


## Overview 🛼
- [x] 点群をクラスタリングする処理
- [ ] 点群Viewer
- [x] 点群のvoxel化処理
- [x] 都市点群からの擬似航空写真作成処理
- [x] [Mask3D](https://github.com/cvg/Mask3D)実行結果を点群に投影する処理
<br><br><br>


## 🎨 点群のクラスタリング処理
#### 実装済みアルゴリズム
- k-means法
- mean-shift
- DBSCAN
- HDBSCAN
<br><br>

#### 実行
以下のように各関数を呼び出して実行します.<br>
clustering.pyのmain文のコメントアウトを解除することによっても利用可能です.<br>
点群データは.pcd形式です.<br>
クラスタ別にBBoxをつける処理、クラスタをBBoxと同じサイズのメッシュに置き換える処理も実装しています.
```
from pcd_algorithm.clustering import Clustering
...

# k-means法を使う場合
clustered_pcds = Clustering().k_means(pcd, <クラスタ数(int)>)

# mean-shiftを使う場合
clustered_pcds = Clustering().mean_shift(pcd, <バンド幅の推定に使用する分位数(float)>, <バンド幅の推定に使用するサンプル数(int)>)

# DBSCANを使う場合
clustered_pcds = Clustering().dbscan(pcd, eps, min_points)

# HDBSCANを使う場合
clustered_pcds = Clustering().hdbscan(pcd, <クラスタを構成するための最小点数(int)>)
```
<br><br><br>


## 🧩 点群のボクセル化処理
点群をローポリにして遊べます.
#### 実行
```
poetry run python pcd_algorithm/voxelize_pcd.py --pcd_path <点群のパス> --voxel_size <ボクセルの一辺の長さ>
```
<br>

#### 実行結果
|入力点群(旅館)|ボクセル化した結果|
|---|---|
|![入力点群(旅館)](https://github.com/sakamo1112/pcd-all/assets/125291665/4b4a8058-83ee-4293-bde0-fcc89cb8f10e)|![ボクセル化した結果](https://github.com/sakamo1112/pcd-all/assets/125291665/edcf8400-4ff3-4107-a935-fc01325c465b)|
<br><br><br>


## 🏙️ 都市点群からの擬似航空写真作成処理
都市点群を擬似的な航空写真に変換します.<br>
ダムモードでは、任意の標高まで水位を上げた場合の点群データとマップを作成することができます.
#### 実行
```
poetry run python pcd_algorithm/create_map.py --pcd_path <点群のパス> --voxel_size <画像上で1pxとする距離> --dam_mode <ダムモードを使いたい場合True> --flood_height <ダムモードで設定する水面の高さ>
```
<br>

#### 実行結果
|入力点群|作成された擬似航空写真|
|---|---|
|![入力点群](https://github.com/sakamo1112/pcd-all/assets/125291665/3c22e0fa-d511-4f89-a398-44bb3cd038ac)|![作成された擬似航空写真](https://github.com/sakamo1112/pcd-all/assets/125291665/32324088-d6fe-417b-8a8b-94964170f3fa)|
|ダム化させた点群|作成された擬似航空写真(ダム化後)|
|![ダム化させた点群](https://github.com/sakamo1112/pcd-all/assets/125291665/2bfc6cc8-45a3-49f3-b34e-d34062223232)|![作成された擬似航空写真(ダム化後)](https://github.com/sakamo1112/pcd-all/assets/125291665/c1c1eae2-2eca-418a-9c2e-6022c69d030a)|
<br><br><br>


## 🎭 Mask3Dの結果描画処理
#### 実行
```
poetry run python pcd_algorithm/Mask3D_mask.py 
--pcd_path <結果を反映させたい点群データ(.ply)のパス> 
--Prefix <Mask3Dで得られた結果をまとめたファイルの格納場所(ディレクトリのパス)> 
--result_file_name <「各オブジェクトについての検出結果を格納したファイルの名前」がまとめられたtxtファイルの名前(パスではない)>
```
（例）各オブジェクトについての検出結果を格納したファイルの名前」の中身
```
pred_mask/20231108_120945_room_0.txt 5 0.9574468731880188
pred_mask/20231108_120945_room_1.txt 16 0.9569514989852905
pred_mask/20231108_120945_room_2.txt 14 0.9293325543403625
pred_mask/20231108_120945_room_3.txt 8 0.9196611642837524
pred_mask/20231108_120945_room_4.txt 10 0.9063308835029602
```
<br>

#### 実行結果
|大学生の部屋の点群|Mask3D🎭適用結果|
|---|---|
|![大学生の部屋の点群](https://github.com/sakamo1112/pcd-all/assets/125291665/6363f1ce-6ec1-4607-8771-d17f7535bbb7)|![Mask3D適用結果](https://github.com/sakamo1112/pcd-all/assets/125291665/0f43c9bb-1cf3-4ac2-b4b6-5b018eb951f4)|

