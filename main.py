import tkinter as tk
import random
import math
from tkinter import messagebox
from collections import defaultdict

# Input parameters
n1, n2, n3, n4 = 4, 1, 3, 2
n = 10 + n3  # 13 vertices
seed = int(f"{n1}{n2}{n3}{n4}")
random.seed(seed)
k = 1.0 - n3 * 0.01 - n4 * 0.005 - 0.05  # Fixed coefficient formula

# Dynamic grid size calculation
grid_size = math.ceil(math.sqrt(n))
ROWS, COLS = grid_size, grid_size

def generate_Adir(n: int, k: float) -> list[list[int]]:
  """Генерує направлену матрицю суміжності"""
  return [
      [0 if random.uniform(0, 2.0) * k < 1.0 else 1 for _ in range(n)]
      for _ in range(n)
  ]

def make_Aundir(Adir: list[list[int]]) -> list[list[int]]:
  """Створює неорієнтовану матрицю з направленої"""
  n = len(Adir)
  return [
      [1 if Adir[i][j] == 1 or Adir[j][i] == 1 else 0 for j in range(n)]
      for i in range(n)
  ]

def generate_weight_matrix(Aundir: list[list[int]]) -> list[list[float]]:
  """Генерує матрицю ваг згідно з алгоритмом завдання"""
  n = len(Aundir)
  
  # Матриця B з випадковими числами
  random.seed(seed)
  B = [[random.uniform(0, 2.0) for _ in range(n)] for _ in range(n)]
  
  # Матриця C
  C = [[math.ceil(B[i][j] * 100 * Aundir[i][j]) for j in range(n)] for i in range(n)]
  
  # Матриця D
  D = [[0 if C[i][j] == 0 else 1 for j in range(n)] for i in range(n)]
  
  # Матриця H
  H = [[1 if D[i][j] != D[j][i] else 0 for j in range(n)] for i in range(n)]
  
  # Матриця Tr (верхня трикутна)
  Tr = [[1 if i < j else 0 for j in range(n)] for i in range(n)]
  
  # Матриця ваг W - виправлена формула
  W = [[0 for _ in range(n)] for _ in range(n)]
  for i in range(n):
      for j in range(i, n):  # Тільки верхній трикутник + діагональ
          W[i][j] = (D[i][j] + H[i][j] * Tr[i][j]) * C[i][j]
          W[j][i] = W[i][j]  # Симетрична матриця
  
  return W

def matrix_to_adjacency_list(matrix: list[list[int]], weights: list[list[float]] = None) -> dict:
  """Конвертує матрицю суміжності в список суміжності (динамічний список)"""
  graph = defaultdict(list)
  n = len(matrix)
  for i in range(n):
      for j in range(n):
          if matrix[i][j] == 1:
              if weights and weights[i][j] > 0:
                  graph[i].append((j, int(weights[i][j])))
              else:
                  graph[i].append(j)
  return graph

def add_vertex(graph: dict, vertex: int) -> None:
  """Додає вершину до графа"""
  if vertex not in graph:
      graph[vertex] = []
      print(f"Додано вершину {vertex}")

def remove_vertex(graph: dict, vertex: int) -> None:
  """Видаляє вершину з графа"""
  if vertex in graph:
      # Видаляємо всі ребра, що ведуть до цієї вершини
      for v in graph:
          graph[v] = [edge for edge in graph[v] if (isinstance(edge, tuple) and edge[0] != vertex) or (isinstance(edge, int) and edge != vertex)]
      # Видаляємо саму вершину
      del graph[vertex]
      print(f"Видалено вершину {vertex}")

def add_edge(graph: dict, u: int, v: int, weight: int = None) -> None:
  """Додає ребро до графа"""
  if u not in graph:
      add_vertex(graph, u)
  if v not in graph:
      add_vertex(graph, v)
  
  if weight is not None:
      if (v, weight) not in graph[u]:
          graph[u].append((v, weight))
          graph[v].append((u, weight))  # Для неорієнтованого графа
          print(f"Додано ребро ({u}, {v}) з вагою {weight}")
  else:
      if v not in graph[u]:
          graph[u].append(v)
          graph[v].append(u)  # Для неорієнтованого графа
          print(f"Додано ребро ({u}, {v})")

def remove_edge(graph: dict, u: int, v: int) -> None:
  """Видаляє ребро з графа"""
  if u in graph:
      graph[u] = [edge for edge in graph[u] if (isinstance(edge, tuple) and edge[0] != v) or (isinstance(edge, int) and edge != v)]
  if v in graph:
      graph[v] = [edge for edge in graph[v] if (isinstance(edge, tuple) and edge[0] != u) or (isinstance(edge, int) and edge != u)]
  print(f"Видалено ребро ({u}, {v})")

def traverse_graph_dfs(graph: dict, start: int, visited: set = None) -> list:
  """Обхід графа в глибину (DFS)"""
  if visited is None:
      visited = set()
  
  path = []
  if start not in visited:
      visited.add(start)
      path.append(start)
      print(f"Відвідуємо вершину {start}")
      
      if start in graph:
          for neighbor in graph[start]:
              if isinstance(neighbor, tuple):
                  neighbor_vertex = neighbor[0]
              else:
                  neighbor_vertex = neighbor
              
              if neighbor_vertex not in visited:
                  path.extend(traverse_graph_dfs(graph, neighbor_vertex, visited))
  
  return path

def print_matrices(Adir: list[list[int]], Aundir: list[list[int]], W: list[list[float]]) -> None:
  """Виводить матриці в термінал"""
  print("="*50)
  print("МАТРИЦІ ГРАФА")
  print("="*50)
  
  print("Directed adjacency matrix (Adir):")
  for i, row in enumerate(Adir):
      print(f"  {i}: {row}")

  print("\nUndirected adjacency matrix (Aundir):")
  for i, row in enumerate(Aundir):
      print(f"  {i}: {row}")
      
  print("\nWeight matrix (W):")
  for i, row in enumerate(W):
      print(f"  {i}: {[int(w) for w in row]}")

def print_edges(Aundir: list[list[int]], W: list[list[float]]) -> None:
  """Виводить всі ребра графа"""
  print("\n" + "="*50)
  print("ВСІ РЕБРА ГРАФА")
  print("="*50)
  
  edges = []
  n = len(Aundir)
  for i in range(n):
      for j in range(i+1, n):  # Тільки верхній трикутник
          if Aundir[i][j] == 1:
              edges.append((i, j, int(W[i][j])))
  
  # Сортуємо ребра за вагою
  edges.sort(key=lambda x: x[2])
  
  for i, (u, v, weight) in enumerate(edges):
      print(f"Ребро {i+1:2d}: ({u:2d}, {v:2d}) з вагою {weight:3d}")
  
  print(f"\nЗагальна кількість ребер: {len(edges)}")

def print_adjacency_list(graph: dict) -> None:
  """Виводить список суміжності"""
  print("\n" + "="*50)
  print("СПИСОК СУМІЖНОСТІ (ДИНАМІЧНИЙ СПИСОК)")
  print("="*50)
  
  for vertex in sorted(graph.keys()):
      neighbors = graph[vertex]
      if neighbors:
          neighbor_str = ", ".join([f"{n[0]}(w:{n[1]})" if isinstance(n, tuple) else str(n) for n in neighbors])
          print(f"Вершина {vertex:2d}: [{neighbor_str}]")
      else:
          print(f"Вершина {vertex:2d}: []")

class DisjointSet:
  """Структура даних для алгоритму Краскала"""
  def __init__(self, n):
      self.parent = list(range(n))
      self.rank = [0] * n
      
  def find(self, x):
      if self.parent[x] != x:
          self.parent[x] = self.find(self.parent[x])
      return self.parent[x]
      
  def union(self, x, y):
      root_x = self.find(x)
      root_y = self.find(y)
      
      if root_x == root_y:
          return False
          
      if self.rank[root_x] < self.rank[root_y]:
          self.parent[root_x] = root_y
      elif self.rank[root_x] > self.rank[root_y]:
          self.parent[root_y] = root_x
      else:
          self.parent[root_y] = root_x
          self.rank[root_x] += 1
          
      return True

class GraphVisualizer:
  def __init__(self, root, Adir, Aundir, W, graph_list):
      self.root = root
      self.canvas_size = 600
      self.margin = 50
      self.node_radius = 15

      self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg='white')
      self.canvas.pack(pady=10)

      self.show_mst = False
      self.mst_edges = []
      self.current_step = 0
      self.algorithm_steps = []
      self.algorithm_type = "kruskal" if n4 % 2 == 0 else "prim"

      # Buttons frame
      frame = tk.Frame(root)
      frame.pack(pady=5)

      self.btn_undirected = tk.Button(frame, text="Неорієнтований граф", command=self.show_undirected)
      self.btn_undirected.pack(side=tk.LEFT, padx=5)
      
      self.btn_mst = tk.Button(frame, text=f"Показати MST ({self.algorithm_type.capitalize()})", command=self.toggle_mst)
      self.btn_mst.pack(side=tk.LEFT, padx=5)
      
      self.btn_step = tk.Button(frame, text="Наступний крок", command=self.next_step, state=tk.DISABLED)
      self.btn_step.pack(side=tk.LEFT, padx=5)
      
      self.btn_reset = tk.Button(frame, text="Скинути", command=self.reset_mst, state=tk.DISABLED)
      self.btn_reset.pack(side=tk.LEFT, padx=5)

      # Graph operations frame
      ops_frame = tk.Frame(root)
      ops_frame.pack(pady=5)
      
      self.btn_traverse = tk.Button(ops_frame, text="Обхід графа (DFS)", command=self.traverse_graph)
      self.btn_traverse.pack(side=tk.LEFT, padx=5)
      
      self.btn_show_list = tk.Button(ops_frame, text="Показати список суміжності", command=self.show_adjacency_list)
      self.btn_show_list.pack(side=tk.LEFT, padx=5)

      self.Adir = Adir
      self.Aundir = Aundir
      self.W = W
      self.graph_list = graph_list

      # Calculate positions for vertices in a grid layout
      self.positions = []
      count = 0
      for r in range(ROWS):
          for c in range(COLS):
              if count < n:
                  x = self.margin + c * ((self.canvas_size - 2 * self.margin) / (COLS - 1))
                  y = self.margin + r * ((self.canvas_size - 2 * self.margin) / (ROWS - 1))
                  self.positions.append((x, y))
                  count += 1

      # Status label
      self.status_var = tk.StringVar()
      self.status_var.set("Готовий до роботи")
      self.status_label = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
      self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

      self.draw_graph()

  def find_obstacles_on_path(self, start_pos, end_pos):
      """Знаходить вершини, які знаходяться близько до прямої лінії"""
      obstacles = []
      
      for i, pos in enumerate(self.positions):
          if pos == start_pos or pos == end_pos:
              continue
              
          x0, y0 = pos
          x1, y1 = start_pos
          x2, y2 = end_pos
          
          if x1 == x2 and y1 == y2:
              continue
              
          A = y2 - y1
          B = x1 - x2
          C = x2 * y1 - x1 * y2
          
          distance = abs(A * x0 + B * y0 + C) / math.sqrt(A * A + B * B)
          
          dot_product = (x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)
          line_length_squared = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
          
          if line_length_squared == 0:
              continue
              
          t = dot_product / line_length_squared
          
          if 0 <= t <= 1 and distance < self.node_radius * 2.5:
              obstacles.append((i, pos, distance))
      
      return obstacles

  def draw_curved_edge(self, start_pos, end_pos, color="black", width=1, arrow=False, is_mst=False):
      """Малює криву лінію між двома точками"""
      x1, y1 = start_pos
      x2, y2 = end_pos
      
      obstacles = self.find_obstacles_on_path(start_pos, end_pos)
      
      if not obstacles:
          dx, dy = x2 - x1, y2 - y1
          dist = math.sqrt(dx * dx + dy * dy)
          if dist == 0:
              return
              
          offset_x = dx / dist * self.node_radius
          offset_y = dy / dist * self.node_radius
          start_x, start_y = x1 + offset_x, y1 + offset_y
          end_x, end_y = x2 - offset_x, y2 - offset_y
          
          if arrow:
              self.canvas.create_line(start_x, start_y, end_x, end_y, 
                                    fill=color, width=width, arrow=tk.LAST, arrowshape=(10, 12, 3))
          else:
              self.canvas.create_line(start_x, start_y, end_x, end_y, fill=color, width=width)
      else:
          self.draw_bezier_curve(start_pos, end_pos, obstacles, color, width, arrow, is_mst)

  def draw_bezier_curve(self, start_pos, end_pos, obstacles, color, width, arrow, is_mst):
      """Малює криву Безьє"""
      x1, y1 = start_pos
      x2, y2 = end_pos
      
      mid_x = (x1 + x2) / 2
      mid_y = (y1 + y2) / 2
      
      dx = x2 - x1
      dy = y2 - y1
      length = math.sqrt(dx * dx + dy * dy)
      
      if length == 0:
          return
          
      perp_x = -dy / length
      perp_y = dx / length
      
      curve_strength = 30
      if obstacles:
          min_distance = min(obs[2] for obs in obstacles)
          curve_strength = max(30, 60 - min_distance * 2)
      
      control1_x = mid_x + perp_x * curve_strength
      control1_y = mid_y + perp_y * curve_strength
      
      points = []
      num_points = 20
      
      for i in range(num_points + 1):
          t = i / num_points
          x = (1-t)**2 * x1 + 2*(1-t)*t * control1_x + t**2 * x2
          y = (1-t)**2 * y1 + 2*(1-t)*t * control1_y + t**2 * y2
          points.extend([x, y])
      
      if len(points) >= 4:
          start_dx = points[2] - points[0]
          start_dy = points[3] - points[1]
          start_dist = math.sqrt(start_dx**2 + start_dy**2)
          if start_dist > 0:
              start_offset = self.node_radius / start_dist
              points[0] += start_dx * start_offset
              points[1] += start_dy * start_offset
          
          end_dx = points[-2] - points[-4]
          end_dy = points[-1] - points[-3]
          end_dist = math.sqrt(end_dx**2 + end_dy**2)
          if end_dist > 0:
              end_offset = self.node_radius / end_dist
              points[-2] -= end_dx * end_offset
              points[-1] -= end_dy * end_offset
      
      if len(points) >= 4:
          if arrow:
              self.canvas.create_line(points, fill=color, width=width, smooth=True)
              if len(points) >= 4:
                  end_x, end_y = points[-2], points[-1]
                  prev_x, prev_y = points[-4], points[-3]
                  arrow_dx = end_x - prev_x
                  arrow_dy = end_y - prev_y
                  arrow_length = math.sqrt(arrow_dx**2 + arrow_dy**2)
                  if arrow_length > 0:
                      arrow_dx /= arrow_length
                      arrow_dy /= arrow_length
                      
                      arrow_size = 8
                      arrow_x1 = end_x - arrow_dx * arrow_size + arrow_dy * arrow_size/2
                      arrow_y1 = end_y - arrow_dy * arrow_size - arrow_dx * arrow_size/2
                      arrow_x2 = end_x - arrow_dx * arrow_size - arrow_dy * arrow_size/2
                      arrow_y2 = end_y - arrow_dy * arrow_size + arrow_dx * arrow_size/2
                      
                      self.canvas.create_polygon([end_x, end_y, arrow_x1, arrow_y1, arrow_x2, arrow_y2], 
                                               fill=color, outline=color)
          else:
              self.canvas.create_line(points, fill=color, width=width, smooth=True)

  def clear_canvas(self):
      self.canvas.delete("all")

  def draw_graph(self):
      self.clear_canvas()
      
      if self.show_mst:
          matrix = self.Aundir
          # Draw MST edges
          for i, j, weight in self.mst_edges[:self.current_step]:
              start_pos = self.positions[i]
              end_pos = self.positions[j]
              
              self.draw_curved_edge(start_pos, end_pos, color="green", width=3, is_mst=True)
              
              mid_x = (start_pos[0] + end_pos[0]) / 2
              mid_y = (start_pos[1] + end_pos[1]) / 2
              self.canvas.create_oval(mid_x-12, mid_y-10, mid_x+12, mid_y+10, fill="white", outline="green", width=2)
              self.canvas.create_text(mid_x, mid_y, text=str(weight), font=("Arial", 9, "bold"), fill="green")
          
          # Draw remaining edges
          for i in range(n):
              for j in range(i+1, n):
                  if matrix[i][j] == 1:
                      if any((i == e[0] and j == e[1]) or (i == e[1] and j == e[0]) for e in self.mst_edges[:self.current_step]):
                          continue
                          
                      start_pos = self.positions[i]
                      end_pos = self.positions[j]
                      
                      self.draw_curved_edge(start_pos, end_pos, color="lightgray", width=1)
                      
                      if self.W[i][j] > 0:
                          mid_x = (start_pos[0] + end_pos[0]) / 2
                          mid_y = (start_pos[1] + end_pos[1]) / 2
                          self.canvas.create_oval(mid_x-10, mid_y-8, mid_x+10, mid_y+8, fill="white", outline="gray")
                          self.canvas.create_text(mid_x, mid_y, text=str(int(self.W[i][j])), font=("Arial", 8))
      else:
          matrix = self.Aundir

          # Draw edges for undirected graph without weights
          for i in range(n):
              for j in range(i+1, n):  # Only upper triangle to avoid duplicates
                  if matrix[i][j] == 1:
                      start_pos = self.positions[i]
                      end_pos = self.positions[j]
                      self.draw_curved_edge(start_pos, end_pos, color="black", width=2)
                      
                      # No weight display for undirected graph

          # Draw self-loops for undirected graph
          for i in range(n):
              if matrix[i][i] == 1:
                  x, y = self.positions[i]
                  r = self.node_radius
                  self.canvas.create_oval(x + r, y - r * 1.5, x - 2*r, y - r * 0.5, outline="black", width=2)

      # Draw nodes
      for i, (x, y) in enumerate(self.positions):
          r = self.node_radius
          self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="lightblue", outline="black", width=2)
          self.canvas.create_text(x, y, text=str(i), font=("Arial", 12, "bold"))

  def show_undirected(self):
      self.show_mst = False
      self.btn_step.config(state=tk.DISABLED)
      self.btn_reset.config(state=tk.DISABLED)
      self.status_var.set("Відображення неорієнтованого графа")
      self.draw_graph()
      
  def toggle_mst(self):
      if not self.show_mst:
          self.show_mst = True
          self.current_step = 0
          self.btn_step.config(state=tk.NORMAL)
          self.btn_reset.config(state=tk.NORMAL)
          
          if self.algorithm_type == "kruskal":
              self.prepare_kruskal_steps()
          else:
              self.prepare_prim_steps()
              
          self.status_var.set(f"Алгоритм {self.algorithm_type.capitalize()}: Крок 0/{len(self.mst_edges)}")
      else:
          self.show_mst = False
          self.btn_step.config(state=tk.DISABLED)
          self.btn_reset.config(state=tk.DISABLED)
          self.status_var.set("Відображення неорієнтованого графа")
          
      self.draw_graph()
      
  def prepare_kruskal_steps(self):
      """Підготовка кроків для алгоритму Краскала"""
      self.mst_edges = []
      edges = []
      
      for i in range(n):
          for j in range(i+1, n):
              if self.Aundir[i][j] == 1:
                  edges.append((i, j, int(self.W[i][j])))
  
      edges.sort(key=lambda x: x[2])
  
      print("\n" + "="*50)
      print("АЛГОРИТМ КРАСКАЛА")
      print("="*50)
      print("Сортовані ребра за вагою:")
      for i, (u, v, w) in enumerate(edges):
          print(f"  {i+1:2d}. ({u:2d}, {v:2d}) - вага {w:3d}")
      print()
  
      ds = DisjointSet(n)
      step = 1
      for u, v, weight in edges:
          if ds.union(u, v):
              self.mst_edges.append((u, v, weight))
              print(f"Крок {step:2d}: Додано ребро ({u:2d}, {v:2d}) з вагою {weight:3d}")
              step += 1
              if len(self.mst_edges) == n - 1:
                  break
          else:
              print(f"Крок {step:2d}: Пропущено ребро ({u:2d}, {v:2d}) з вагою {weight:3d} (утворює цикл)")
              step += 1
  
      total_weight = sum(weight for _, _, weight in self.mst_edges)
      print(f"\nЗагальна вага MST: {total_weight}")
      print("="*50)
      self.algorithm_steps = self.mst_edges.copy()
      
  def prepare_prim_steps(self):
      """Підготовка кроків для алгоритму Прима"""
      self.mst_edges = []
  
      graph = defaultdict(list)
      for i in range(n):
          for j in range(n):
              if self.Aundir[i][j] == 1:
                  graph[i].append((j, int(self.W[i][j])))
  
      print("\n" + "="*50)
      print("АЛГОРИТМ ПРИМА")
      print("="*50)
  
      start_vertex = 0
      visited = [False] * n
      visited[start_vertex] = True
      print(f"Початкова вершина: {start_vertex}")
      print()
  
      def find_min_edge():
          min_weight = float('inf')
          min_edge = None
      
          for u in range(n):
              if visited[u]:
                  for v, weight in graph[u]:
                      if not visited[v] and weight < min_weight:
                          min_weight = weight
                          min_edge = (u, v, weight)
      
          return min_edge
  
      step = 1
      for _ in range(n-1):
          edge = find_min_edge()
          if edge:
              u, v, weight = edge
              self.mst_edges.append((u, v, weight))
              visited[v] = True
              print(f"Крок {step:2d}: Додано ребро ({u:2d}, {v:2d}) з вагою {weight:3d}")
              step += 1
          else:
              print("Граф не зв'язний, MST неможливо побудувати")
              break
  
      total_weight = sum(weight for _, _, weight in self.mst_edges)
      print(f"\nЗагальна вага MST: {total_weight}")
      print("="*50)
      self.algorithm_steps = self.mst_edges.copy()
      
  def next_step(self):
      """Перехід до наступного кроку алгоритму"""
      if self.current_step < len(self.mst_edges):
          self.current_step += 1
          self.status_var.set(f"Алгоритм {self.algorithm_type.capitalize()}: Крок {self.current_step}/{len(self.mst_edges)}")
          self.draw_graph()
          
          if self.current_step > 0:
              u, v, weight = self.mst_edges[self.current_step - 1]
              messagebox.showinfo("Крок алгоритму", 
                                 f"Додано ребро ({u}, {v}) з вагою {weight}")
      
  def reset_mst(self):
      """Скидання візуалізації MST"""
      self.current_step = 0
      self.status_var.set(f"Алгоритм {self.algorithm_type.capitalize()}: Крок 0/{len(self.mst_edges)}")
      self.draw_graph()

  def traverse_graph(self):
      """Обхід графа в глибину"""
      print("\n" + "="*50)
      print("ОБХІД ГРАФА В ГЛИБИНУ (DFS)")
      print("="*50)
      
      visited_path = traverse_graph_dfs(self.graph_list, 0)
      print(f"Шлях обходу: {' -> '.join(map(str, visited_path))}")
      print("="*50)
      
      messagebox.showinfo("Обхід графа", f"Шлях DFS: {' -> '.join(map(str, visited_path))}")

  def show_adjacency_list(self):
      """Показує список суміжності в терміналі"""
      print_adjacency_list(self.graph_list)

def main():
  print("ЛАБОРАТОРНА РОБОТА 6 - МІНІМАЛЬНІ КІСТЯКИ")
  print("="*60)
  print(f"Параметри: n1={n1}, n2={n2}, n3={n3}, n4={n4}")
  print(f"Кількість вершин: {n}")
  print(f"Коефіцієнт k: {k:.3f}")
  print(f"Seed: {seed}")
  print(f"Алгоритм: {'Краскала' if n4 % 2 == 0 else 'Прима'} (n4 {'парне' if n4 % 2 == 0 else 'непарне'})")
  
  # Генерація матриць
  Adir = generate_Adir(n, k)
  Aundir = make_Aundir(Adir)
  W = generate_weight_matrix(Aundir)
  
  # Виведення матриць
  print_matrices(Adir, Aundir, W)
  print_edges(Aundir, W)
  
  # Створення списку суміжності
  graph_list = matrix_to_adjacency_list(Aundir, W)
  print_adjacency_list(graph_list)
  
  # Демонстрація функцій роботи з графом
  print("\n" + "="*50)
  print("ДЕМОНСТРАЦІЯ ФУНКЦІЙ РОБОТИ З ГРАФОМ")
  print("="*50)
  
  # Створення GUI
  root = tk.Tk()
  root.title("Лабораторна робота 6 - Мінімальні кістяки")
  root.geometry("800x750")
  
  # Інформаційна панель
  info_frame = tk.Frame(root)
  info_frame.pack(pady=5)
  
  info_text = f"Параметри: n1={n1}, n2={n2}, n3={n3}, n4={n4} | Вершин: {n} | k={k:.3f}"
  info_label = tk.Label(info_frame, text=info_text, font=("Arial", 10))
  info_label.pack()
  
  algorithm_text = f"Алгоритм: {'Краскала' if n4 % 2 == 0 else 'Прима'} (n4={'парне' if n4 % 2 == 0 else 'непарне'})"
  algorithm_label = tk.Label(info_frame, text=algorithm_text, font=("Arial", 10, "bold"))
  algorithm_label.pack()
  
  app = GraphVisualizer(root, Adir, Aundir, W, graph_list)
  root.mainloop()

if __name__ == "__main__":
  main()
