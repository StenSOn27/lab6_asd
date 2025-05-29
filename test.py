import math
import random
import tkinter as tk
from typing import List, Tuple

Coord = Tuple[float, float]
IntMatrix = List[List[int]]
FloatMatrix = List[List[float]]
Edge = Tuple[int, int, int]  # u, v, w

#### Configuration
VARIANT: int = 4107
n1, n2, n3, n4 = map(int, str(VARIANT).zfill(4))
####

VERTEX_CNT = 10 + n3
SEED = VARIANT
PANEL_SIZE = 600
PANEL_GAP = 40
OUTER_RADIUS = 0.40 * PANEL_SIZE
NODE_R = 22
EDGE_W = 3


def generate_directed_matrix(n: int) -> IntMatrix:
    random.seed(SEED)
    k = 1.0 - n3 * 0.01 - n4 * 0.005 - 0.05
    return [
        [1 if random.uniform(0.0, 2.0) * k >= 1.0 else 0 for _ in range(n)]
        for _ in range(n)
    ]


def to_undirected(mat: IntMatrix) -> IntMatrix:
    n = len(mat)
    res = [row[:] for row in mat]
    for i in range(n):
        for j in range(i + 1, n):
            res[i][j] = res[j][i] = 1 if mat[i][j] or mat[j][i] else 0
    return res


def generate_weight_matrix(adj: IntMatrix) -> IntMatrix:
    random.seed(SEED)
    n = len(adj)
    b: FloatMatrix = [[random.uniform(0.0, 2.0) for _ in range(n)] for _ in range(n)]
    c: IntMatrix = [[math.ceil(b[i][j] * 100 * adj[i][j]) for j in range(n)] for i in range(n)]
    d: IntMatrix = [[0 if c[i][j] == 0 else 1 for j in range(n)] for i in range(n)]
    h: IntMatrix = [[1 if d[i][j] != d[j][i] else 0 for j in range(n)] for i in range(n)]
    tr: IntMatrix = [[1 if i < j else 0 for j in range(n)] for i in range(n)]
    weights: IntMatrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            w = (d[i][j] + h[i][j] + tr[i][j]) * c[i][j]
            weights[i][j] = weights[j][i] = w
    return weights


def prim_mst(weights: IntMatrix) -> Tuple[List[Edge], int]:
    n = len(weights)
    in_tree = [False] * n
    dist = [math.inf] * n
    parent = [-1] * n

    dist[0] = 0
    total = 0
    edges: List[Edge] = []

    for _ in range(n):
        u = min((d, v) for v, d in enumerate(dist) if not in_tree[v])[1]
        in_tree[u] = True
        if parent[u] != -1:
            edges.append((parent[u], u, weights[u][parent[u]]))
            total += weights[u][parent[u]]

        for v in range(n):
            w = weights[u][v]
            if not in_tree[v] and 0 < w < dist[v]:
                dist[v] = w
                parent[v] = u
    return edges, total


def node_positions(n: int, center: int, ox: int) -> List[Coord]:
    cx, cy = ox + PANEL_SIZE / 2, PANEL_SIZE / 2
    pos = [None] * n  # type: ignore
    pos[center] = (cx, cy)
    outer = [i for i in range(n) if i != center]
    for k, i in enumerate(outer):
        ang = 2 * math.pi * k / len(outer)
        pos[i] = (cx + OUTER_RADIUS * math.cos(ang),
                  cy + OUTER_RADIUS * math.sin(ang))
    return pos


def shift(p: Coord, q: Coord, d: float) -> Coord:
    dx, dy = q[0] - p[0], q[1] - p[1]
    l = math.hypot(dx, dy)
    return (p[0] + dx / l * d, p[1] + dy / l * d)


def draw_node(cv: tk.Canvas, x: float, y: float, lbl: str):
    cv.create_oval(x - NODE_R, y - NODE_R, x + NODE_R, y + NODE_R,
                   fill="coral", width=2)
    cv.create_text(x, y, text=lbl, font=("Arial", 12, "bold"))


def draw_weight_label(cv: tk.Canvas, x: float, y: float, w: int):
    text_id = cv.create_text(x, y, text=str(w), fill="black", font=("Arial", 13))
    bbox = cv.bbox(text_id)
    if bbox:
        x1, y1, x2, y2 = bbox
        pad = 3
        rect_id = cv.create_rectangle(x1 - pad, y1 - pad, x2 + pad, y2 + pad,
                                      fill="white", outline="black")
        cv.tag_lower(rect_id, text_id)


def draw_edge(cv: tk.Canvas, p: Coord, q: Coord, w: int,
              color: str = "black", width: int = EDGE_W):
    a = shift(p, q, NODE_R)
    b = shift(q, p, NODE_R)

    options = {"fill": color, "width": width, "capstyle": tk.ROUND}
    line_id = cv.create_line(*a, *b, **options)

    cv.tag_lower(line_id)

    mid = (
        a[0] + 0.5 * (b[0] - a[0]),
        a[1] + 0.5 * (b[1] - a[1])
    )
    draw_weight_label(cv, mid[0], mid[1], w)


def print_matrix(mat: IntMatrix, title: str):
    n = len(mat)
    print(f"\n{title}")
    print("   " + " ".join(f"{j + 1:>3}" for j in range(n)))
    print("   " + "---" * n)
    for i in range(n):
        row = " ".join(f"{mat[i][j]:>3}" for j in range(n))
        print(f"{i + 1:>2}| {row}")


def print_edge_list(edges: List[Edge], title: str):
    print(f"\n{title}")
    for k, (u, v, w) in enumerate(edges, 1):
        print(f"Ребро {u}-{v}, вага: {w}")


def get_all_edges(weights: IntMatrix) -> List[Edge]:
    n = len(weights)
    return sorted([(i, j, weights[i][j]) for i in range(n) for j in range(i + 1, n) if weights[i][j] > 0],
                  key=lambda e: e[2])


def main():
    directed = generate_directed_matrix(VERTEX_CNT)
    undirected = to_undirected(directed)
    weights = generate_weight_matrix(undirected)

    mst_edges, mst_weight = prim_mst(weights)
    all_edges = get_all_edges(weights)

    print_matrix(undirected, "Adjacency matrix (undirected)")
    print_matrix(weights, "Weight matrix W")
    print_edge_list(all_edges, "Усі ребра графа (за зростанням ваги)")
    print_edge_list(mst_edges, "Ребра мінімального кістяка (Прима)")

    print(f"\nАлгоритм: Прима")
    print("Ребра мін. кістяка:", [(u + 1, v + 1, w) for u, v, w in mst_edges])
    print("Сума ваг ребер МК =", mst_weight)

    cv_w = 2 * PANEL_SIZE + PANEL_GAP
    root = tk.Tk()
    root.title(f"Lab 6 · Variant {VARIANT}")
    cv = tk.Canvas(root, width=cv_w, height=PANEL_SIZE + 20, bg="white")
    cv.pack()

    pos_L = node_positions(VERTEX_CNT, VERTEX_CNT // 2, 0)
    pos_R = node_positions(VERTEX_CNT, VERTEX_CNT // 2, PANEL_SIZE + PANEL_GAP)

    # повний граф
    for i in range(VERTEX_CNT):
        for j in range(i + 1, VERTEX_CNT):
            if weights[i][j] > 0:
                draw_edge(cv, pos_L[i], pos_L[j], weights[i][j])
    for i, (x, y) in enumerate(pos_L):
        draw_node(cv, x, y, str(i + 1))
    cv.create_text(PANEL_SIZE / 2, PANEL_SIZE - 10,
                   text="Граф із вагами", font=("Arial", 15, "bold"))

    # мінімальний кістяк
    for u, v, w in mst_edges:
        draw_edge(cv, pos_R[u], pos_R[v], w, color="green", width=EDGE_W + 1)
    for i, (x, y) in enumerate(pos_R):
        draw_node(cv, x, y, str(i + 1))
    cv.create_text(PANEL_SIZE + PANEL_GAP + PANEL_SIZE / 2,
                   PANEL_SIZE - 10,
                   text=f"Мін. кістяк (алгоритм Прима)\nСума ваг: {mst_weight}",
                   font=("Arial", 15, "bold"), fill="green")

    root.mainloop()


if __name__ == "__main__":
    main()