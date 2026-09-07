import json
import csv
import re
import subprocess
import sys
from collections import deque

MAX_DEPTH = 3


# ==========================================================
# 1. Shell command 실행
# ==========================================================
def run_command(cmd):
    result = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True
    )
    return result.stdout


# ==========================================================
# 2. Package name 정규화
# ==========================================================
def normalize_package(name):
    """
    APT 출력에 포함되는 architecture qualifier 등을 제거한다.

    예:
        apache2-utils:i386 -> apache2-utils
        perl:any           -> perl
    """

    name = name.strip()

    # <perl:any>, <debconf-2.0> 등 가상 패키지는
    # 실제 패키지 노드로 사용하지 않음
    if name.startswith("<") and name.endswith(">"):
        return None

    # architecture qualifier 제거
    if ":" in name:
        name = name.split(":")[0]

    return name


# ==========================================================
# 3. OBOM 로드
# ==========================================================
def load_obom(obom_file):
    with open(obom_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    components = data.get("components", [])

    component_map = {}

    for component in components:
        name = component.get("name")

        if not name:
            continue

        normalized_name = normalize_package(name)

        if not normalized_name:
            continue

        component_info = {
            "name": normalized_name,
            "version": component.get("version", ""),
            "type": component.get("type", ""),
            "purl": component.get("purl", ""),
            "bom_ref": component.get("bom-ref", "")
        }

        # 동일 name이 여러 번 존재할 경우
        # Debian package PURL을 우선 사용
        if normalized_name not in component_map:
            component_map[normalized_name] = component_info

        else:
            current = component_map[normalized_name]

            new_purl = component_info.get("purl", "")
            current_purl = current.get("purl", "")

            if (
                new_purl.startswith("pkg:deb/")
                and not current_purl.startswith("pkg:deb/")
            ):
                component_map[normalized_name] = component_info

    return component_map, len(components)


# ==========================================================
# 4. APT dependency 추출
# ==========================================================
def get_dependencies(package):
    """
    apt-cache depends 결과에서
    Depends / PreDepends 관계만 추출한다.

    Recommends / Suggests / Conflicts / Replaces 제외
    """

    output = run_command(
        f"apt-cache depends {package}"
    )

    dependencies = set()

    for line in output.splitlines():

        line = line.strip()

        # |Depends: 처리
        line = line.lstrip("|")

        match = re.match(
            r"(Depends|PreDepends):\s+(.+)",
            line
        )

        if not match:
            continue

        dep = match.group(2).strip()

        dep = normalize_package(dep)

        if dep:
            dependencies.add(dep)

    return sorted(dependencies)


# ==========================================================
# 5. 실제 설치 여부 확인
# ==========================================================
def is_installed(package):
    output = run_command(
        f"dpkg-query -W -f='${{Status}}' '{package}'"
    )

    return "install ok installed" in output


# ==========================================================
# 6. Dependency Graph 생성
# ==========================================================
def build_graph(root_package, component_map, max_depth):

    nodes = {}
    edges = set()

    queue = deque()

    queue.append(
        (root_package, 0)
    )

    # Root component 확인
    if root_package not in component_map:
        print(
            f"[!] Root package '{root_package}' "
            f"not found in OBOM"
        )
        sys.exit(1)

    root_info = component_map[root_package]

    nodes[root_package] = {
        **root_info,
        "depth": 0,
        "direct": 0
    }

    visited_depth = {
        root_package: 0
    }

    while queue:

        package, depth = queue.popleft()

        if depth >= max_depth:
            continue

        dependencies = get_dependencies(package)

        for dep in dependencies:

            next_depth = depth + 1

            # ------------------------------------------
            # 실제 설치 여부 확인
            # ------------------------------------------
            if not is_installed(dep):
                continue

            # ------------------------------------------
            # OBOM 존재 여부 확인
            # ------------------------------------------
            if dep not in component_map:
                continue

            # ------------------------------------------
            # Edge 저장
            # ------------------------------------------
            edges.add(
                (package, dep)
            )

            # ------------------------------------------
            # Node 저장
            # ------------------------------------------
            if dep not in nodes:

                component_info = component_map[dep]

                nodes[dep] = {
                    **component_info,
                    "depth": next_depth,
                    "direct": (
                        1 if next_depth == 1 else 0
                    )
                }

                visited_depth[dep] = next_depth

                queue.append(
                    (dep, next_depth)
                )

            else:

                # 더 짧은 경로 발견
                if next_depth < nodes[dep]["depth"]:

                    nodes[dep]["depth"] = next_depth

                    nodes[dep]["direct"] = (
                        1 if next_depth == 1 else 0
                    )

                    visited_depth[dep] = next_depth

                    queue.append(
                        (dep, next_depth)
                    )

    return nodes, edges


# ==========================================================
# 7. nodes.csv 저장
# ==========================================================
def save_nodes(nodes):

    sorted_nodes = sorted(
        nodes.items(),
        key=lambda x: (
            x[1]["depth"],
            x[0]
        )
    )

    with open(
        "nodes.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "node_id",
            "name",
            "version",
            "type",
            "purl",
            "depth",
            "direct"
        ])

        for node_id, (name, info) in enumerate(
            sorted_nodes
        ):

            writer.writerow([
                node_id,
                name,
                info.get("version", ""),
                info.get("type", ""),
                info.get("purl", ""),
                info.get("depth", ""),
                info.get("direct", "")
            ])


# ==========================================================
# 8. edges.csv 저장
# ==========================================================
def save_edges(edges):

    with open(
        "edges.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "source",
            "target"
        ])

        for source, target in sorted(edges):

            writer.writerow([
                source,
                target
            ])


# ==========================================================
# 9. 결과 출력
# ==========================================================
def print_summary(
    root_package,
    total_components,
    nodes,
    edges
):

    print()
    print("========================================")
    print(" Dependency Mapping Result")
    print("========================================")

    print(
        f"[+] OBOM Components : {total_components}"
    )

    print(
        f"[+] Root Package    : {root_package}"
    )

    root_info = nodes[root_package]

    print(
        f"[+] Version         : "
        f"{root_info.get('version', '')}"
    )

    print(
        f"[+] Type            : "
        f"{root_info.get('type', '')}"
    )

    print(
        f"[+] PURL            : "
        f"{root_info.get('purl', '')}"
    )

    print()
    print(
        f"[+] Matched Nodes   : {len(nodes)}"
    )

    print(
        f"[+] Dependency Edges: {len(edges)}"
    )

    print()

    depth_count = {}

    for info in nodes.values():

        depth = info["depth"]

        depth_count[depth] = (
            depth_count.get(depth, 0) + 1
        )

    print("[+] Depth Statistics")

    for depth in sorted(depth_count):

        print(
            f"    {depth}-Hop : "
            f"{depth_count[depth]}"
        )

    print()
    print("[+] Generated Files")
    print("    nodes.csv")
    print("    edges.csv")


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    if len(sys.argv) < 3:

        print(
            "Usage:"
        )

        print(
            "python3 dependency_mapper.py "
            "<obom.json> <root-package>"
        )

        print()
        print(
            "Example:"
        )

        print(
            "python3 dependency_mapper.py "
            "ubuntu-obom.json apache2"
        )

        sys.exit(1)

    obom_file = sys.argv[1]
    root_package = sys.argv[2]

    print(
        f"[+] Loading OBOM: {obom_file}"
    )

    component_map, total_components = load_obom(
        obom_file
    )

    print(
        f"[+] Total OBOM Components: "
        f"{total_components}"
    )

    print(
        f"[+] Root Package: {root_package}"
    )

    print(
        f"[+] Max Depth: {MAX_DEPTH}"
    )

    nodes, edges = build_graph(
        root_package,
        component_map,
        MAX_DEPTH
    )

    save_nodes(nodes)
    save_edges(edges)

    print_summary(
        root_package,
        total_components,
        nodes,
        edges
    )
