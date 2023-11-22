# 🌈 pcd_all 🌈
点群を扱う処理であれば何でも上げるリポジトリです.
<br><br>


## Overview 🛼
- [x] 点群をクラスタリングする処理
- [ ] 点群Viewer
- [x] 点群のvoxel化処理
- [x] [Mask3D](https://github.com/cvg/Mask3D)実行結果を点群に投影する処理
<br><br><br>

### 🎨 点群のクラスタリング処理
---
#### 実装済みアルゴリズム
- k-means法
- mean-shift
- DBSCAN
- HDBSCAN
<br><br>

#### 実行
以下のように各関数を呼び出して実行します.<br>
clustering.pyのmain文のコメントアウトを解除することによっても利用可能です.<br>
点群データは.pcd形式です.
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
<br><br>


### 🎭 Mask3Dの結果描画処理
---
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

