import argparse
import csv
import json
import os
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GAE, GCNConv
from torch_geometric.utils import to_undirected


# ==========================================
# 0. 재현성 설정
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ==========================================
# 1. 입력 데이터 로드
# ==========================================
def load_graph_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_networkx_graph(data_dict):
    """
    Edge 방향은 기존 코드와 동일하게 유지한다.
    source -> target = source가 target에 의존함 (dependent -> dependency)
    """
    G = nx.DiGraph()

    for node in data_dict["nodes"]:
        node_id = node["id"]
        G.add_node(node_id, **node)

    for edge in data_dict["edges"]:
        src = edge["source"]
        dst = edge["target"]
        if src in G and dst in G:
            G.add_edge(src, dst)

    return G


# ==========================================
# 2. Patch Impact Score(PIS) 계산
# ==========================================
def calculate_weighted_impact(G, node, decay=0.7):
    """
    I_i = sum(lambda^(d_ki - 1))

    - ancestors: 대상 컴포넌트에 직접/간접적으로 의존하는 모든 상위 컴포넌트
    - distance : 상위 컴포넌트 k에서 대상 i까지의 최단 의존거리
    - 1-Hop = 1.0, 2-Hop = lambda, 3-Hop = lambda^2 ...
    """
    ancestors = nx.ancestors(G, node)
    impact = 0.0

    for parent in ancestors:
        try:
            distance = nx.shortest_path_length(G, source=parent, target=node)
            impact += decay ** (distance - 1)
        except nx.NetworkXNoPath:
            continue

    return impact


def calculate_pis(G, decay=0.7):
    results = []

    for node in G.nodes():
        impact = calculate_weighted_impact(G, node, decay=decay)
        results.append({
            "node_id": node,
            "weighted_impact": impact,
            "parent_count": G.in_degree(node),
            "dependency_count": G.out_degree(node),
            "transitive_parent_count": len(nx.ancestors(G, node)),
            "degree_centrality": nx.degree_centrality(G)[node],
        })

    impacts = [row["weighted_impact"] for row in results]
    min_impact = min(impacts) if impacts else 0.0
    max_impact = max(impacts) if impacts else 0.0

    for row in results:
        if max_impact == min_impact:
            row["pis"] = 0.0
        else:
            row["pis"] = 10.0 * (
                (row["weighted_impact"] - min_impact)
                / (max_impact - min_impact)
            )

    return results


# ==========================================
# 3. GCN 입력 Feature 생성
# ==========================================
def calculate_depths(G, root_node):
    """root_node로부터의 최단거리(Depth)를 계산한다."""
    if root_node not in G:
        raise ValueError(f"Root node '{root_node}'가 그래프에 없습니다.")

    depths = {node: None for node in G.nodes()}
    lengths = nx.single_source_shortest_path_length(G, root_node)

    for node, depth in lengths.items():
        depths[node] = depth

    # 혹시 root에서 도달하지 못하는 노드가 있다면 마지막 depth + 1로 처리
    finite_depths = [d for d in depths.values() if d is not None]
    fallback_depth = (max(finite_depths) + 1) if finite_depths else 0

    for node in depths:
        if depths[node] is None:
            depths[node] = fallback_depth

    return depths


def build_pyg_data(G, root_node):
    """
    초기 One-hot 대신 연구 방법론과 맞는 구조적 Feature를 사용한다.

    X_i = [Direct, Depth, Parent Count, Dependency Count]
    """
    node_ids = list(G.nodes())
    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    depths = calculate_depths(G, root_node)

    features = []
    for node in node_ids:
        depth = depths[node]
        direct = 1.0 if depth == 1 else 0.0
        parent_count = float(G.in_degree(node))
        dependency_count = float(G.out_degree(node))

        features.append([
            direct,
            float(depth),
            parent_count,
            dependency_count,
        ])

    # Feature scale 차이를 줄이기 위해 표준화
    scaler = StandardScaler()
    x_np = scaler.fit_transform(np.asarray(features, dtype=np.float32))
    x = torch.tensor(x_np, dtype=torch.float)

    edge_list = []
    for src, dst in G.edges():
        edge_list.append([id_to_idx[src], id_to_idx[dst]])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # GCN/GAE 임베딩은 구조적 이웃 표현을 학습하는 보조 시각화 목적이므로
    # 학습 시에는 양방향 메시지 패싱을 위해 undirected edge_index를 사용한다.
    train_edge_index = to_undirected(edge_index)

    data = Data(x=x, edge_index=train_edge_index)
    return data, node_ids, id_to_idx, depths


# ==========================================
# 4. GCN 기반 Graph Autoencoder
# ==========================================
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=32, out_channels=16):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def train_gcn_autoencoder(data, epochs=300, learning_rate=0.01):
    """
    Label이 없는 환경에서 adjacency reconstruction을 이용한 비지도 학습.
    GCN은 PIS를 예측하지 않고 구조적 임베딩 생성에만 사용한다.
    """
    encoder = GCNEncoder(
        in_channels=data.num_features,
        hidden_channels=32,
        out_channels=16,
    )
    model = GAE(encoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"[GCN/GAE] Epoch {epoch:03d} | Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data.x, data.edge_index).cpu().numpy()

    return embeddings


# ==========================================
# 5. t-SNE 2차원 축소
# ==========================================
def run_tsne(node_embeddings):
    num_nodes = node_embeddings.shape[0]

    if num_nodes < 3:
        raise ValueError("t-SNE 시각화를 위해서는 최소 3개 이상의 노드가 필요합니다.")

    # perplexity는 반드시 num_nodes보다 작아야 함
    perplexity = min(30, max(5, num_nodes // 3), num_nodes - 1)

    tsne = TSNE(
        n_components=2,
        random_state=SEED,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(node_embeddings)


# ==========================================
# 6. 결과 저장
# ==========================================
def save_pis_results(results, output_path):
    fieldnames = [
        "node_id",
        "weighted_impact",
        "pis",
        "parent_count",
        "transitive_parent_count",
        "dependency_count",
        "degree_centrality",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in sorted(results, key=lambda x: x["pis"], reverse=True):
            writer.writerow({
                "node_id": row["node_id"],
                "weighted_impact": f'{row["weighted_impact"]:.4f}',
                "pis": f'{row["pis"]:.4f}',
                "parent_count": row["parent_count"],
                "transitive_parent_count": row["transitive_parent_count"],
                "dependency_count": row["dependency_count"],
                "degree_centrality": f'{row["degree_centrality"]:.6f}',
            })


def save_embedding_results(node_ids, embeddings_2d, pis_map, depths, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "tsne_x", "tsne_y", "pis", "depth"])

        for idx, node_id in enumerate(node_ids):
            writer.writerow([
                node_id,
                f"{embeddings_2d[idx, 0]:.6f}",
                f"{embeddings_2d[idx, 1]:.6f}",
                f"{pis_map[node_id]:.4f}",
                depths[node_id],
            ])


# ==========================================
# 7. 시각화 1: 전체 의존성 그래프
# ==========================================
def plot_overall_graph(G, pos, output_path):
    plt.figure(figsize=(16, 11))

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=320,
        node_color="skyblue",
        alpha=0.90,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="->",
        arrowsize=9,
        edge_color="gray",
        alpha=0.35,
        width=0.8,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=7,
        font_family="sans-serif",
    )

    plt.title("GCN/t-SNE-based Overall Dependency Graph Visualization", fontsize=15)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ==========================================
# 8. 시각화 2: PIS Top-10 강조 그래프
# ==========================================
def plot_pis_top10_graph(G, pos, pis_results, root_node, output_path):
    sorted_results = sorted(pis_results, key=lambda x: x["pis"], reverse=True)
    top10_results = sorted_results[:10]
    top10_nodes = [row["node_id"] for row in top10_results]
    top10_set = set(top10_nodes)

    base_nodes = [node for node in G.nodes() if node not in top10_set]

    plt.figure(figsize=(16, 11))

    # 전체 dependency edge는 유지하되 연하게 표시
    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="->",
        arrowsize=8,
        edge_color="lightgray",
        alpha=0.28,
        width=0.7,
    )

    # 비 Top-10 노드: 구조적 배경 역할
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=base_nodes,
        node_size=150,
        node_color="lightgray",
        alpha=0.45,
    )

    # PIS Top-10: PIS에 비례하여 크기와 색상 강조
    top10_pis = np.array([row["pis"] for row in top10_results], dtype=float)
    top10_sizes = 500 + (top10_pis * 120)

    if top10_pis.max() == top10_pis.min():
        normalized = np.ones_like(top10_pis)
    else:
        normalized = (top10_pis - top10_pis.min()) / (top10_pis.max() - top10_pis.min())

    top10_colors = plt.cm.YlOrRd(0.35 + 0.60 * normalized)

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=top10_nodes,
        node_size=top10_sizes,
        node_color=top10_colors,
        alpha=0.95,
        edgecolors="black",
        linewidths=0.8,
    )

    # PIS Top-10만 라벨 표시
    labels = {
        row["node_id"]: f'{row["node_id"]}\nPIS={row["pis"]:.2f}'
        for row in top10_results
    }

    # root가 Top-10에 없더라도 분석 기준점임을 표시
    if root_node in G and root_node not in labels:
        labels[root_node] = f"{root_node}\n(root)"

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=8,
        font_family="sans-serif",
        font_weight="bold",
    )

    plt.title("PIS Top-10 Highlighted Dependency Graph", fontsize=15)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ==========================================
# 9. 실행
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="GCN/t-SNE dependency visualization with PIS Top-10 highlighting"
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        default="apache2-dependency-with-cve.json",
        help="Dependency graph JSON path",
    )
    parser.add_argument(
        "--root",
        default="apache2",
        help="Root package used to calculate Depth (default: apache2)",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.7,
        help="PIS distance decay factor lambda (default: 0.7)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="GCN/GAE training epochs (default: 300)",
    )
    parser.add_argument(
        "--output-dir",
        default="result",
        help="Output directory (default: result)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(" PIS + GCN/t-SNE Dependency Graph Visualization")
    print("=" * 70)

    # 1) JSON -> NetworkX directed graph
    data_dict = load_graph_json(args.json_path)
    G = build_networkx_graph(data_dict)

    print(f"[+] Nodes : {G.number_of_nodes()}")
    print(f"[+] Edges : {G.number_of_edges()}")
    print(f"[+] Root  : {args.root}")
    print(f"[+] Lambda: {args.decay}")

    # 2) PIS 계산
    pis_results = calculate_pis(G, decay=args.decay)
    pis_map = {row["node_id"]: row["pis"] for row in pis_results}

    print("\n[+] PIS Top-10")
    for rank, row in enumerate(
        sorted(pis_results, key=lambda x: x["pis"], reverse=True)[:10],
        start=1,
    ):
        print(
            f"    {rank:2d}. {row['node_id']:<25} "
            f"PIS={row['pis']:.4f}  impact={row['weighted_impact']:.4f}"
        )

    # 3) 구조 Feature -> GCN/GAE 학습 -> Embedding
    pyg_data, node_ids, _, depths = build_pyg_data(G, args.root)
    node_embeddings = train_gcn_autoencoder(
        pyg_data,
        epochs=args.epochs,
        learning_rate=0.01,
    )

    # 4) t-SNE: 한 번만 계산하고 두 그림에서 같은 좌표 사용
    embeddings_2d = run_tsne(node_embeddings)
    pos = {
        node_id: (embeddings_2d[idx, 0], embeddings_2d[idx, 1])
        for idx, node_id in enumerate(node_ids)
    }

    # 5) 결과 파일 저장
    pis_csv = os.path.join(args.output_dir, "pis_results.csv")
    embedding_csv = os.path.join(args.output_dir, "gcn_tsne_embeddings.csv")
    overall_png = os.path.join(args.output_dir, "dependency_graph_overall.png")
    top10_png = os.path.join(args.output_dir, "dependency_graph_pis_top10.png")

    save_pis_results(pis_results, pis_csv)
    save_embedding_results(node_ids, embeddings_2d, pis_map, depths, embedding_csv)
    plot_overall_graph(G, pos, overall_png)
    plot_pis_top10_graph(G, pos, pis_results, args.root, top10_png)

    print("\n[+] Saved")
    print(f"    - {pis_csv}")
    print(f"    - {embedding_csv}")
    print(f"    - {overall_png}")
    print(f"    - {top10_png}")
    print("\n[+] 완료: 두 그래프는 동일한 t-SNE 좌표를 사용합니다.")


if __name__ == "__main__":
    main()
