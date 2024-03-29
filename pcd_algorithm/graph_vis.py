import argparse

import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
import pandas as pd  # type: ignore
import pandera as pa  # type: ignore
import polars as pl


class Schema(pa.DataFrameModel):
    edge_number: int = pa.Field(ge=0)
    node_start: int = pa.Field(ge=0)
    node_end: int = pa.Field(ge=0)


def draw_graph(G: nx.Graph) -> None:
    """
    6つのレイアウトでグラフを描画する。

    Parameters:
    G (nx.Graph): 描画するグラフ
    """
    fig, axs = plt.subplots(2, 3, figsize=(10, 6.5))
    axs = axs.flatten()[:5]
    layouts = {
        "Spring": nx.spring_layout,
        "Kamada-Kawai": nx.kamada_kawai_layout,
        "Circular": nx.circular_layout,
        "Random": nx.random_layout,
        "Spectral": nx.spectral_layout,
        "Planar": nx.planar_layout,
    }
    for ax, (title, layout) in zip(axs, layouts.items()):
        pos = layout(G)
        centrality = nx.degree_centrality(G)  # 次数中心性の計算
        node_color = [centrality[node] for node in G.nodes]
        print(node_color)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=node_color,
            cmap="viridis",
            edge_color="k",
            node_size=50,
            font_size=5,
            ax=ax,
        )
        ax.set_title(title)
    plt.suptitle("Graph Visualization with Different Layouts")
    plt.show()

    return None


def draw_graph_3d(G: nx.Graph) -> None:
    """
    グラフを3Dで描画する。

    Parameters:
    G (nx.Graph): 描画するグラフ
    """
    # 3Dのspring layoutを計算
    pos = nx.kamada_kawai_layout(G, dim=3)

    # 3Dプロットの準備
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    centrality = nx.degree_centrality(G)  # 次数中心性の計算
    node_color = [centrality[node] for node in G.nodes]  # 色の強度を調整

    for key, value in pos.items():
        ax.text(value[0], value[1], value[2], f'{key}', color='black', fontsize=3)

    xs, ys, zs = [], [], []
    for key, value in pos.items():
        xs.append(value[0])
        ys.append(value[1])
        zs.append(value[2])
    ax.scatter(xs, ys, zs, c=node_color, cmap='viridis', depthshade=True)

    # エッジを描画
    for edge in G.edges():
        x = [pos[edge[0]][0], pos[edge[1]][0]]
        y = [pos[edge[0]][1], pos[edge[1]][1]]
        z = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x, y, z, color="tab:gray")

    plt.show()


def main(xlsx_path: str, sheet_name: str) -> pd.DataFrame:
    """
    指定されたsheetのデータを抽出し、DataFrameを作成します。

    Parameters:
    data (pd.DataFrame): 全データが格納されているDataFrame
    sheet_name (str): 抽出するイオンの名前

    Returns:
    pd.DataFrame: 抽出されたデータが格納されたDataFrame
    """
    data = pl.read_excel(xlsx_path, sheet_name=sheet_name).to_pandas()
    #data = Schema.validate(data)

    G = nx.Graph()
    for i in range(len(data.index)):
        edge_name = (
            "Edge"
            + str(i)
            + " ("
            + str(data.iloc[i, 1])
            + "-"
            + str(data.iloc[i, 2])
            + ")"
        )
        shop_vector = list(data.iloc[i, 3:])
        G.add_edge(
            data.iloc[i, 1],
            data.iloc[i, 2],
            label="",
            name=edge_name,
            vector_attribute=shop_vector,
        )

    draw_graph(G)
    draw_graph_3d(G)

    return G


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and display a point cloud")
    parser.add_argument(
        "--xlsx_path",
        "-i",
        type=str,
        default="data/aeon_graph.xlsx",
        help="excel file (.xlsx)",
    )
    parser.add_argument("--sheet_name", "-n", type=str, help="sheet name")
    args = parser.parse_args()

    data = main(args.xlsx_path, args.sheet_name)
