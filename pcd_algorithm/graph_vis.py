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
    axs = axs.flatten()[:6]
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
            font_size=2,
            ax=ax,
        )
        ax.set_title(title)
    plt.suptitle("Graph Visualization with Different Layouts")
    plt.show()

    return None


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
    data = Schema.validate(data)

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

    return G


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and display a point cloud")
    parser.add_argument(
        "--xlsx_path",
        "-i",
        type=str,
        default="data/data_for_test.xlsx",
        help="excel file (.xlsx)",
    )
    parser.add_argument("--sheet_name", "-n", type=str, help="sheet name")
    args = parser.parse_args()

    data = main(args.xlsx_path, args.sheet_name)
