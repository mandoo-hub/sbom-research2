# 🕸️ SBOM-based Software Dependency & Patch Impact Visualization

> SBOM/OBOM과 실제 패키지 의존관계를 기반으로 소프트웨어 의존성 그래프를 구축하고,  
> **Patch Impact Score (PIS)** 및 **Graph Convolutional Network (GCN)** 기반 임베딩을 활용하여  
> 소프트웨어 구성요소의 구조적 패치 영향도를 분석하고 시각화하는 프로젝트입니다.

---

## 📌 Project Overview

현대 소프트웨어 환경은 하나의 애플리케이션이 다수의 라이브러리 및 패키지에 의존하는 복잡한 구조를 가지고 있습니다.  
특정 구성요소를 패치하거나 변경할 경우 직접적인 의존관계뿐만 아니라 여러 단계의 전이적 의존관계를 통해 다른 구성요소에도 구조적인 영향을 미칠 수 있습니다.

본 프로젝트에서는 **CycloneDX 기반 OBOM**과 Ubuntu의 실제 패키지 의존정보를 결합하여 방향성 소프트웨어 의존성 그래프를 구축합니다.

전체 분석 과정은 다음과 같습니다.

```text
Software Environment
        ↓
OBOM Generation
        ↓
Dependency Extraction
        ↓
Dependency Graph
        ↓
Structural Analysis
        ├── Patch Impact Score (PIS)
        │
        └── GCN / Graph Autoencoder
                    ↓
             Node Embedding
                    ↓
                  t-SNE
                    ↓
        Dependency Visualization
```

의존성 그래프에서는 다음과 같이 Edge를 정의합니다.

```text
source → target
```

즉,

```text
apache2 → apache2-bin
```

은 `apache2`가 `apache2-bin`에 의존한다는 것을 의미합니다.

---

## 🔎 1. OBOM Generation

먼저 분석 대상 Linux 환경에서 CycloneDX 기반 OBOM을 생성합니다.

본 프로젝트에서는 `cdxgen`을 사용합니다.

```bash
sudo cdxgen -t os -o ubuntu-obom.json .
```

생성된 `ubuntu-obom.json`에는 운영환경에서 식별된 소프트웨어 구성요소의 이름, 버전, 타입, PURL 등의 정보가 포함됩니다.

예시:

```json
{
  "name": "apache2",
  "version": "2.4.52-1ubuntu4.23",
  "type": "library",
  "purl": "pkg:deb/ubuntu/apache2@2.4.52-1ubuntu4.23#var/lib/dpkg"
}
```

---

## 🔗 2. Dependency Graph Generation

OBOM 생성 후 `depends_mapper_v2.py`를 사용하여 분석 대상 패키지의 실제 의존관계를 추출합니다.

### 실행 방법

```bash
python3 depends_mapper_v2.py <OBOM_FILE> <ROOT_PACKAGE>
```

Apache2를 분석하는 경우:

```bash
python3 depends_mapper_v2.py ubuntu-obom.json apache2
```

`depends_mapper_v2.py`는 다음 정보를 결합하여 의존성 그래프를 생성합니다.

- OBOM에 존재하는 소프트웨어 구성요소
- `apt-cache depends` 기반 패키지 의존관계
- 현재 시스템에 실제 설치된 패키지
- `Depends` 및 `PreDepends` 관계

현재 기본 최대 탐색 깊이는 **3-Hop**입니다.

### 실행 결과 예시

```text
========================================
 Dependency Mapping Result
========================================
[+] OBOM Components : 4499
[+] Root Package    : apache2
[+] Version         : 2.4.52-1ubuntu4.23
[+] Type            : library
[+] PURL            : pkg:deb/ubuntu/apache2@2.4.52-1ubuntu4.23#var/lib/dpkg

[+] Matched Nodes   : 52
[+] Dependency Edges: 96

[+] Depth Statistics
    0-Hop : 1
    1-Hop : 7
    2-Hop : 23
    3-Hop : 21

[+] Generated Files
    nodes.csv
    edges.csv
    apache2-dependency.json
```

### 생성 파일

#### `nodes.csv`

분석 대상 그래프에 포함된 소프트웨어 구성요소 정보를 저장합니다.

```text
node_id
name
version
type
purl
depth
direct
```

#### `edges.csv`

패키지 간 의존관계를 저장합니다.

```text
source,target
apache2,apache2-bin
apache2-bin,libc6
...
```

#### `apache2-dependency.json`

PIS 분석 및 GCN 시각화에서 사용하는 최종 Dependency Graph 데이터입니다.

```json
{
  "root": "apache2",
  "nodes": [
    {
      "id": "apache2",
      "name": "apache2",
      "version": "2.4.52-1ubuntu4.23",
      "type": "library",
      "depth": 0,
      "direct": 0
    }
  ],
  "edges": [
    {
      "source": "apache2",
      "target": "apache2-bin"
    }
  ]
}
```

---

## 📐 3. Patch Impact Score (PIS)

본 프로젝트에서는 특정 소프트웨어 구성요소의 변경이 의존성 구조상 어느 범위까지 영향을 미칠 수 있는지를 분석하기 위해 **Patch Impact Score (PIS)**를 사용합니다.

구성요소 \(i\)의 거리 가중 영향도는 다음과 같이 계산합니다.

\[
I_i =
\sum_{k \in \mathcal{P}(i)}
\lambda^{d_{ki}-1}
\]

- \(i\): 분석 대상 구성요소
- \(\mathcal{P}(i)\): 해당 구성요소에 직접 또는 간접적으로 의존하는 상위 구성요소 집합
- \(d_{ki}\): 상위 구성요소 \(k\)에서 대상 구성요소 \(i\)까지의 최단 의존거리
- \(\lambda\): 거리 감쇄 계수 (기본값 `0.7`)

따라서 기본 설정에서는 다음과 같은 가중치를 적용합니다.

```text
1-Hop → 1.00
2-Hop → 0.70
3-Hop → 0.49
```

계산된 영향도는 0~10 범위로 정규화하여 최종 PIS를 산출합니다.

\[
PIS_i =
10
\frac{I_i-I_{min}}
{I_{max}-I_{min}}
\]

> **Note:** PIS는 실제 패치 실패 또는 서비스 장애 발생 확률을 의미하지 않습니다.  
> 현재 구현에서는 추출된 소프트웨어 의존성 그래프를 기반으로 구성요소의 **상대적인 구조적 패치 영향도**를 나타냅니다.

---

## 🧠 4. GCN-based Graph Embedding

`main_pis_visualization.py`에서는 **Graph Convolutional Network (GCN)** 기반 Graph Autoencoder를 이용하여 소프트웨어 구성요소의 구조적 표현을 학습합니다.

GCN의 초기 Node Feature는 다음과 같이 구성합니다.

```text
[Direct, Depth, Parent Count, Dependency Count]
```

- **Direct**: Root Package의 직접 의존성 여부
- **Depth**: Root Package로부터의 최단 의존거리
- **Parent Count**: 해당 구성요소에 직접 의존하는 구성요소 수
- **Dependency Count**: 해당 구성요소가 직접 의존하는 구성요소 수

PIS는 GCN의 입력 Feature로 사용하지 않습니다.

```text
Dependency Graph
       │
       ├──────────────→ PIS Calculation
       │
       └──────────────→ GCN / GAE
                              ↓
                       Node Embedding
                              ↓
                            t-SNE
                              ↓
                       2D Visualization
                              ↑
                         PIS Highlight
```

따라서 PIS와 GCN 임베딩은 동일한 의존성 그래프를 기반으로 독립적으로 계산되며, 최종 시각화 단계에서 PIS가 높은 구성요소를 강조하는 방식으로 결합됩니다.

---

## 📊 5. Visualization

### 실행 방법

`depends_mapper_v2.py`에서 생성한 Dependency JSON 파일을 입력으로 사용합니다.

```bash
python3 main_pis_visualization.py <DEPENDENCY_JSON>
```

Apache2의 경우:

```bash
python3 main_pis_visualization.py apache2-dependency.json
```

기본 설정은 다음과 같습니다.

```text
Root Package : apache2
Decay Factor : 0.7
GCN Epochs   : 300
Output       : ./result/
```

### 생성 결과

```text
result/
├── pis_results.csv
├── gcn_tsne_embeddings.csv
├── dependency_graph_overall.png
└── dependency_graph_pis_top10.png
```

각 파일의 역할은 다음과 같습니다.

| File | Description |
|---|---|
| `pis_results.csv` | 구성요소별 Weighted Impact, PIS 및 구조적 지표 |
| `gcn_tsne_embeddings.csv` | GCN 임베딩을 t-SNE로 축소한 2차원 좌표 |
| `dependency_graph_overall.png` | 전체 소프트웨어 의존성 구조 |
| `dependency_graph_pis_top10.png` | 동일한 구조에서 PIS Top-10 구성요소를 강조한 그래프 |

---

## 📊 Visualization Result

### Overall Dependency Graph

| GCN/t-SNE-based Overall Dependency Graph |
| :---: |
| ![Overall Dependency Graph](./result/dependency_graph_overall.png) |

전체 그래프는 GCN/GAE를 통해 학습한 Node Embedding을 t-SNE를 이용하여 2차원 공간으로 축소한 결과입니다.

### PIS Top-10 Visualization

| PIS Top-10 Highlighted Dependency Graph |
| :---: |
| ![PIS Top-10 Dependency Graph](./result/dependency_graph_pis_top10.png) |

두 번째 그래프는 **동일한 t-SNE 좌표와 전체 의존성 구조를 유지하면서 PIS가 높은 상위 10개 구성요소를 강조**합니다.

이를 통해 전체 소프트웨어 의존성 구조에서 구조적 패치 영향도가 높은 구성요소가 어느 위치에 존재하는지 비교할 수 있습니다.

---

## 🔍 Key Findings & Graph Analysis

Apache2 기반 예비 실험에서는 총 **52개의 구성요소와 96개의 의존관계**가 식별되었습니다.

구조적 지표와 PIS를 비교한 결과 다음과 같은 특징이 확인되었습니다.

1. **`libc6`**
   - Direct Parent Count와 Transitive Parent Count가 모두 높게 나타났으며 PIS에서도 가장 높은 값을 보였습니다.
   - 다수의 구성요소가 직접 또는 간접적으로 의존하는 핵심 공유 라이브러리로 나타났습니다.

2. **`libgcc-s1`**
   - 직접 연결 정도만으로는 두드러지지 않지만 Transitive Parent Count와 PIS에서 높은 순위를 보였습니다.
   - 이는 직접 연결뿐만 아니라 전이적 의존관계와 의존거리를 함께 분석할 필요가 있음을 보여줍니다.

3. **`apache2-bin`**
   - Degree Centrality는 높게 나타났지만 PIS에서는 상대적으로 높은 순위를 보이지 않았습니다.
   - 이는 전체 연결성이 높은 구성요소와 패치 변경 시 상위 구성요소로 구조적 영향이 전파될 가능성이 높은 구성요소가 반드시 동일하지 않을 수 있음을 보여줍니다.

> 본 분석은 실제 패치 장애 데이터를 Ground Truth로 사용한 예측 정확도 평가가 아니라, 소프트웨어 의존성 그래프에서 서로 다른 구조적 특성을 비교하기 위한 예비 분석입니다.

---

## 🛠️ Tech Stack & Architecture

- **Language:** Python 3
- **SBOM / OBOM:** CycloneDX, cdxgen
- **Dependency Extraction:** APT (`apt-cache depends`)
- **Deep Learning / GNN:** PyTorch, PyTorch Geometric
- **Graph Model:** GCN / Graph Autoencoder (GAE)
- **Dimensionality Reduction:** scikit-learn (`t-SNE`)
- **Graph Analysis:** NetworkX
- **Visualization:** Matplotlib
- **Data Format:** CSV, JSON

---

## 📂 Project Structure

```text
.
├── depends_mapper_v2.py             # OBOM 기반 패키지 의존성 추출
├── main_pis_visualization.py        # PIS 계산 및 GCN/t-SNE 시각화
│
├── ubuntu-obom.json                 # cdxgen으로 생성한 원본 OBOM
├── nodes.csv                        # 추출된 구성요소 정보
├── edges.csv                        # 패키지 의존관계
├── apache2-dependency.json          # 분석용 Dependency Graph
│
├── result/
│   ├── pis_results.csv              # 구성요소별 PIS 분석 결과
│   ├── gcn_tsne_embeddings.csv      # t-SNE 2D 좌표
│   ├── dependency_graph_overall.png # 전체 의존성 그래프
│   └── dependency_graph_pis_top10.png # PIS Top-10 강조 그래프
│
├── README.md
└── requirements.txt
```

---

## 🚀 Quick Start

### Step 1. OBOM 생성

```bash
sudo cdxgen -t os -o ubuntu-obom.json .
```

### Step 2. Apache2 Dependency Graph 생성

```bash
python3 depends_mapper_v2.py ubuntu-obom.json apache2
```

생성 결과:

```text
nodes.csv
edges.csv
apache2-dependency.json
```

### Step 3. PIS 분석 및 GCN/t-SNE 시각화

```bash
python3 main_pis_visualization.py apache2-dependency.json
```

### Step 4. 결과 확인

```bash
ls -l result/
```

```text
pis_results.csv
gcn_tsne_embeddings.csv
dependency_graph_overall.png
dependency_graph_pis_top10.png
```

---

## ⚠️ Current Limitations

현재 구현은 Apache2를 Root Package로 설정한 **3-Hop 의존성 그래프 기반 예비 실험**입니다.

PIS는 실제 서비스 장애 확률을 예측하는 모델이 아니라 추출된 의존성 그래프에서 직접·간접 의존관계와 의존거리를 기반으로 계산한 상대적 구조 영향도입니다.

또한 현재 APT Dependency Mapping에서는 `Depends`와 `PreDepends`를 중심으로 의존관계를 구성하며, Virtual Package 및 Alternative Dependency에 대해서는 제한적인 처리를 수행합니다.

향후 연구에서는 Apache2뿐만 아니라 Nginx, Tomcat, MySQL 등 다양한 서비스로 분석 범위를 확대하고, 실제 패치 이력 및 서비스 중요도 등의 정보를 결합하여 분석 방법을 확장할 예정입니다.

---

## 📚 Research Purpose

본 프로젝트의 목적은 기존의 CVSS, EPSS, KEV 등 **취약점 위험도 기반 패치 우선순위 분석을 대체하는 것이 아닙니다.**

기존 접근이 주로 다음 질문을 다룬다면,

> **"어떤 취약점을 먼저 조치해야 하는가?"**

본 연구는 다음과 같은 보완적인 질문에 초점을 둡니다.

> **"이 소프트웨어 구성요소를 패치하거나 변경할 경우, 의존성 구조상 어느 범위까지 영향을 미칠 수 있는가?"**

이를 통해 실제 운영환경에서 패치를 적용하기 전에 소프트웨어 구성요소 간의 구조적 영향범위를 검토할 수 있는 의사결정 지원 방법을 제시하는 것을 목표로 합니다.
