import csv
import networkx as nx

NODES_FILE = "nodes.csv"
EDGES_FILE = "edges.csv"
OUTPUT_FILE = "graph_metrics.csv"


# ==========================================================
# 1. Node 정보 로드
# ==========================================================
def load_nodes():
    nodes = {}

    with open(NODES_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            name = row["name"]

            nodes[name] = {
                "node_id": row["node_id"],
                "version": row["version"],
                "type": row["type"],
                "purl": row["purl"],
                "depth": int(row["depth"]),
                "direct": int(row["direct"])
            }

    return nodes


# ==========================================================
# 2. Edge 정보 로드
# ==========================================================
def load_edges():
    edges = []

    with open(EDGES_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            source = row["source"]
            target = row["target"]

            edges.append((source, target))

    return edges


# ==========================================================
# 3. Dependency Graph 생성
# ==========================================================
def build_graph(nodes, edges):

    G = nx.DiGraph()

    # Node 추가
    for name, info in nodes.items():

        G.add_node(
            name,
            version=info["version"],
            type=info["type"],
            depth=info["depth"],
            direct=info["direct"],
            purl=info["purl"]
        )

    # Edge 추가
    for source, target in edges:
        G.add_edge(source, target)

    return G


# ==========================================================
# 4. Graph Metrics 계산
# ==========================================================
def calculate_metrics(G, nodes):

    # 전체 Degree Centrality
    degree_centrality = nx.degree_centrality(G)

    # In / Out Degree Centrality
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)

    results = []

    for node in G.nodes():

        # 해당 노드로 들어오는 Edge 수
        parent_count = G.in_degree(node)

        # 해당 노드가 의존하는 하위 Component 수
        dependency_count = G.out_degree(node)

        info = nodes[node]

        results.append({
            "name": node,
            "version": info["version"],
            "depth": info["depth"],
            "direct": info["direct"],
            "parent_count": parent_count,
            "dependency_count": dependency_count,
            "degree_centrality": degree_centrality[node],
            "in_degree_centrality": in_degree_centrality[node],
            "out_degree_centrality": out_degree_centrality[node]
        })

    return results


# ==========================================================
# 5. 결과 CSV 저장
# ==========================================================
def save_results(results):

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        fieldnames = [
            "name",
            "version",
            "depth",
            "direct",
            "parent_count",
            "dependency_count",
            "degree_centrality",
            "in_degree_centrality",
            "out_degree_centrality"
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in results:

            row["degree_centrality"] = round(
                row["degree_centrality"], 6
            )

            row["in_degree_centrality"] = round(
                row["in_degree_centrality"], 6
            )

            row["out_degree_centrality"] = round(
                row["out_degree_centrality"], 6
            )

            writer.writerow(row)


# ==========================================================
# 6. 결과 요약 출력
# ==========================================================
def print_summary(G, results):

    print()
    print("========================================")
    print(" Dependency Graph Analysis")
    print("========================================")

    print(f"[+] Nodes : {G.number_of_nodes()}")
    print(f"[+] Edges : {G.number_of_edges()}")

    print()
    print("[+] Top 10 - Parent Count")

    parent_rank = sorted(
        results,
        key=lambda x: x["parent_count"],
        reverse=True
    )

    for rank, row in enumerate(parent_rank[:10], 1):

        print(
            f"{rank:2}. "
            f"{row['name']:<25} "
            f"parents={row['parent_count']}"
        )

    print()
    print("[+] Top 10 - Degree Centrality")

    centrality_rank = sorted(
        results,
        key=lambda x: x["degree_centrality"],
        reverse=True
    )

    for rank, row in enumerate(
        centrality_rank[:10],
        1
    ):

        print(
            f"{rank:2}. "
            f"{row['name']:<25} "
            f"centrality="
            f"{row['degree_centrality']:.6f}"
        )

    print()
    print(f"[+] Result saved : {OUTPUT_FILE}")


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    nodes = load_nodes()
    edges = load_edges()

    G = build_graph(
        nodes,
        edges
    )

    results = calculate_metrics(
        G,
        nodes
    )

    save_results(results)

    print_summary(
        G,
        results
    )
