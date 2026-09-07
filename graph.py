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
        
        transitive_parents = nx.ancestors(G, node)
        
        transitive_parent_count = len(transitive_parents)

        weighted_impact = calculate_weighted_impact(G, node, decay=0.7)

        info = nodes[node]

        results.append({
            "name": node,
            "version": info["version"],
            "depth": info["depth"],
            "direct": info["direct"],
            "parent_count": parent_count,
            "transitive_parent_count" : transitive_parent_count,
            "weighted_impact": weighted_impact,
            "dependency_count": dependency_count,
            "degree_centrality": degree_centrality[node],
            "in_degree_centrality": in_degree_centrality[node],
            "out_degree_centrality": out_degree_centrality[node]
        })

    return results
    
    
# ==========================================================
# 5. 감쇄 계수  계산
# ==========================================================
def calculate_weighted_impact(G, node, decay=0.7):
    
    ancestors = nx.ancestors(G, node)
    
    impact = 0.0
    
    for parent in ancestors:
    
      try:
          distance = nx.shortest_path_length(
              G,
              source=parent,
              target=node
          )
          
          impact += decay ** (distance - 1)
          
      except nx.NetworkXNoPath:
          continue
      
    return impact


# ==========================================================
# 6. PIS  계산
# ==========================================================
def calculate_pis(results):

    impacts = [
        row["weighted_impact"]
        for row in results
    ]
    
    min_impact = min(impacts)
    max_impact = max(impacts)
    
    for row in results:
        
        if max_impact == min_impact:
            pis = 0.0
            
        else:
            pis = (
                10
                * (
                    row["weighted_impact"]
                    - min_impact
                  )
                  / (
                      max_impact
                      - min_impact
                    )
                  )
                  
            row["pis"] = pis
            
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
            "transitive_parent_count",
            "weighted_impact",
            "pis",
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
    print("[++] Top 10 - Parent Count")

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
    print("[+] Top 10 - Transitive Parent Count")
    
    transitive_rank = sorted(
        results,
        key=lambda x: x["transitive_parent_count"],
        reverse=True
    )
    
    for rank, row in enumerate(
        transitive_rank[:10],
        1
    ):
        print(
            f"{rank:2}. "
            f"{row['name']:<25} "
            f"transitive_parents="
            f"{row['transitive_parent_count']}"
    )

    print()
    print("[+] Top 10 - Patch Impact Score")
    
    pis_rank = sorted(
        results,
        key=lambda x: x["pis"],
        reverse=True
    )
    
    for rank, row in enumerate(
        pis_rank[:10],
        1
    ):
    
        print(
            f"{rank:2}. "
            f"{row['name']:<25} "
            f"PIS={row['pis']:.4f} "
            f"impact={row['weighted_impact']:.4f}"
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

    results = calculate_pis(
        results
    )


    save_results(results)

    print_summary(
        G,
        results
    )