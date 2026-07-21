import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ==============================
# 0. CARGA Y PREPROCESAMIENTO
# ==============================
print("=" * 70)
print("  ARBOLES DE DEPENDENCIA E INFORMACION MUTUA")
print("  Student Dropout (Realinho et al., 2021)")
print("=" * 70)

df = pd.read_csv('data.csv', sep=';')
df.columns = df.columns.str.strip()
print(f"\n>>> Dataset original: {df.shape[0]} filas x {df.shape[1]} columnas")

# Seleccion manual de 9 variables
VARS = {
    'Admission grade': 'x1',
    'Age at enrollment': 'x2',
    'Tuition fees up to date': 'x3',
    'Scholarship holder': 'x4',
    'Curricular units 1st sem (approved)': 'x5',
    'Curricular units 1st sem (grade)': 'x6',
    'Curricular units 2nd sem (approved)': 'x7',
    'Curricular units 2nd sem (grade)': 'x8',
    'Marital status': 'x9',
}

df_work = df[list(VARS.keys()) + ['Target', 'Displaced']].copy()
df_work.rename(columns=VARS, inplace=True)
df_work.rename(columns={'Target': 'Target'}, inplace=True)

# Target a numerico
target_map = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
df_work['Target'] = df_work['Target'].map(target_map)

print(f">>> Variables: {list(VARS.values())}")
print(f">>> Target: Target (0=Dropout, 1=Enrolled, 2=Graduate)")
print(f">>> Split: Displaced")

# ==============================
# 1. DISCRETIZACION
# ==============================
print("\n>>> Discretizando variables continuas...")

for col in ['x1', 'x2', 'x5', 'x6', 'x7', 'x8']:
    try:
        df_work[col] = pd.qcut(df_work[col], q=3, labels=[0, 1, 2], duplicates='drop')
    except:
        med = df_work[col].median()
        df_work[col] = (df_work[col] > med).astype(int)

# x9 (marital status): 1=soltero, resto=otro
df_work['x9'] = df_work['x9'].map(lambda x: 0 if x == 1 else 1)

print(">>> Discretizacion completada.")

# Dataset original (sin split) para comparativa
df_orig = df_work.drop(columns=['Displaced']).copy()

# ==============================
# 2. SPLIT POR Displaced
# ==============================
df_A_full = df_work[df_work['Displaced'] == 0].copy()
df_B_full = df_work[df_work['Displaced'] == 1].copy()

# Eliminar variable split
df_A = df_A_full.drop(columns=['Displaced'])
df_B = df_B_full.drop(columns=['Displaced'])

print(f"\n>>> Dataset Original:    {df_orig.shape[0]} filas")
print(f">>> Dataset A (No Desp):  {df_A.shape[0]} filas")
print(f">>> Dataset B (Desp):     {df_B.shape[0]} filas")

COLS = list(df_A.columns)
print(f">>> Columnas: {COLS}")

# ==============================
# 3. FUNCIONES
# ==============================
def entropia(col):
    probs = col.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs))

def entropia_detallada(col, nombre):
    conteo = col.value_counts().sort_index()
    total = len(col)
    print(f"\n  --- {nombre} ---")
    print(f"  Valores y frecuencias:")
    H = 0
    terminos = []
    for v in conteo.index:
        p = conteo[v] / total
        logp = np.log2(p)
        term = -p * logp
        H += term
        terminos.append((v, conteo[v], p, logp, term))
        print(f"    p({v}) = {conteo[v]:>4}/{total} = {p:.4f}")
    print(f"  Calculo:")
    for _, _, p, logp, term in terminos:
        print(f"    -({p:.4f} * log2({p:.4f})) = -({p:.4f} * {logp:.4f}) = {term:.4f}")
    print(f"  H({nombre}) = {' + '.join([f'{t[4]:.4f}' for t in terminos])} = {H:.4f}")
    return H

def mi_detallada(datos, col1, col2):
    probs_conj = datos.groupby([col1, col2]).size() / len(datos)
    p_x = datos[col1].value_counts(normalize=True)
    p_y = datos[col2].value_counts(normalize=True)

    print(f"\n  --- IM({col1}, {col2}) ---")
    print(f"  Probabilidades conjuntas y terminos:")
    mi_total = 0.0
    terminos = []
    for (v_x, v_y), p_xy in probs_conj.items():
        p_xv = p_x[v_x]
        p_yv = p_y[v_y]
        ratio = p_xy / (p_xv * p_yv)
        log_ratio = np.log2(ratio)
        term = p_xy * log_ratio
        mi_total += term
        terminos.append((v_x, v_y, p_xy, p_xv, p_yv, ratio, log_ratio, term))
        signo = '+' if term > 0 else '-'
        print(f"    p({col1}={v_x}, {col2}={v_y})={p_xy:.4f}  "
              f"p({col1})={p_xv:.4f}  p({col2})={p_yv:.4f}  "
              f"ratio={ratio:.4f}  log2={log_ratio:.4f}  term={term:+.4f}")

    print(f"\n  IM({col1}, {col2}) = {' + '.join([f'({t[7]:+.4f})' for t in terminos])} = {mi_total:.4f}")
    print(f"  Verificacion sklearn: {mutual_info_score(datos[col1], datos[col2]) / np.log(2):.4f}")
    return mi_total

def matriz_mi(datos):
    cols = datos.columns
    n = len(cols)
    mi_mat = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                mi_mat.iloc[i, j] = 0.0
            else:
                mi = mutual_info_score(datos.iloc[:, i], datos.iloc[:, j]) / np.log(2)
                mi_mat.iloc[i, j] = mi
                mi_mat.iloc[j, i] = mi
    return mi_mat

def kruskal_detallado(mi_mat):
    cols = list(mi_mat.columns)
    n = len(cols)
    aristas_todas = []
    for i in range(n):
        for j in range(i + 1, n):
            aristas_todas.append((cols[i], cols[j], mi_mat.iloc[i, j]))
    aristas_todas.sort(key=lambda x: x[2], reverse=True)

    padre = {c: c for c in cols}
    def find(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            padre[ra] = rb
            return True
        return False

    aceptadas = []
    rechazadas = []
    for u, v, w in aristas_todas:
        if len(aceptadas) == n - 1:
            break
        if union(u, v):
            aceptadas.append((u, v, w))
            print(f"    Aceptada:  {u} -- {v}  (peso: {w:.4f})")
        else:
            rechazadas.append((u, v, w))
            print(f"    Rechazada: {u} -- {v}  (peso: {w:.4f}) [ciclo]")

    peso_total = sum(w for _, _, w in aceptadas)
    return aceptadas, rechazadas, peso_total

def prim_detallado(mi_mat, inicio):
    cols = list(mi_mat.columns)
    visitados = {inicio}
    frontera = []
    for v in cols:
        if v != inicio:
            w = mi_mat.loc[inicio, v]
            frontera.append((w, inicio, v))
    frontera.sort(key=lambda x: x[0], reverse=True)

    aristas = []
    peso_total = 0.0
    paso = 1

    while len(visitados) < len(cols):
        mejor = None
        for peso, u, v in frontera:
            if v not in visitados:
                mejor = (peso, u, v)
                break
        if mejor is None:
            break
        peso, u, v = mejor
        visitados.add(v)
        aristas.append((u, v, peso))
        peso_total += peso
        print(f"    Paso {paso}: {u} -- {v}  (peso: {peso:.4f})")
        paso += 1
        for w in cols:
            if w not in visitados:
                pw = mi_mat.loc[v, w]
                frontera.append((pw, v, w))
        frontera.sort(key=lambda x: x[0], reverse=True)
    return aristas, peso_total

def graficar_arbol(aristas, nombre, titulo):
    G = nx.Graph()
    G.add_weighted_edges_from(aristas)
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=2.5, iterations=200)
    nx.draw_networkx_nodes(G, pos, node_size=2800, node_color='#87CEFA', edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G, pos, width=2.5, edge_color='#555555')
    labels = {(u, v): f'{d["weight"]:.4f}' for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=9, font_color='red')
    plt.title(titulo, fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(nombre, format='PNG', dpi=300, bbox_inches='tight')
    plt.close()
    print(f">>> {nombre}")

def graficar_arbol_circular(aristas, nombre, titulo):
    G = nx.Graph()
    G.add_weighted_edges_from(aristas)
    plt.figure(figsize=(13, 13))
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=2800, node_color='#87CEFA', edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold')
    nx.draw_networkx_edges(G, pos, width=2.5, edge_color='#555555')
    labels = {(u, v): f'{d["weight"]:.4f}' for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=9, font_color='red')
    plt.title(titulo, fontsize=16, fontweight='bold', pad=30)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(nombre, format='PNG', dpi=300, bbox_inches='tight')
    plt.close()
    print(f">>> {nombre}")

# ==============================
# 4. PROCESAMIENTO COMPLETO
# ==============================
def procesar_dataset(datos, nombre, nodo_inicio):
    cols = list(datos.columns)
    print(f"\n{'='*70}")
    print(f"  DATASET {nombre}  ({datos.shape[0]} filas)")
    print(f"{'='*70}")

    print(f"\n--- ENTROPIAS ---")
    entropias = {}
    for c in cols:
        h = entropia_detallada(datos[c], c)
        entropias[c] = h

    print(f"\n--- EJEMPLO DETALLADO DE IM ---")
    mi_detallada(datos, cols[-2], cols[-1])

    print(f"\n--- MATRIZ DE IM ---")
    mi_mat = matriz_mi(datos)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 20)
    print(mi_mat.round(4).to_string())

    print(f"\n--- ALGORITMO DE KRUSKAL (MST) ---")
    aristas, rechazadas, peso = kruskal_detallado(mi_mat)
    print(f"  Peso total MST = {peso:.4f}")

    print(f"\n--- VALIDACION POR PRIM (desde {nodo_inicio}) ---")
    aristas_prim, peso_prim = prim_detallado(mi_mat, nodo_inicio)
    print(f"  Peso total Prim = {peso_prim:.4f}")
    print(f"  Coinciden: {abs(peso - peso_prim) < 1e-6}")

    graficar_arbol(aristas, f"arbol_{nombre}.png",
                   f"Arbol de Dependencias MI - Student Dropout\nDataset {nombre}")
    graficar_arbol_circular(aristas, f"arbol_{nombre}_circular.png",
                            f"Arbol de Dependencias MI - Student Dropout\nDataset {nombre} (Circular)")

    return entropias, mi_mat, aristas, peso

# Datasets
E_ORIG, MI_ORIG, ARISTAS_ORIG, PESO_ORIG = procesar_dataset(df_orig, 'Original', 'Target')
E_A, MI_A, ARISTAS_A, PESO_A = procesar_dataset(df_A, 'A_NoDesplazados', 'Target')
E_B, MI_B, ARISTAS_B, PESO_B = procesar_dataset(df_B, 'B_Desplazados', 'Target')

# ==============================
# 5. COMPARATIVA
# ==============================
print(f"\n{'='*70}")
print(f"  COMPARATIVA DE ENTROPIAS")
print(f"{'='*70}")
print(f"{'Variable':<10} {'H(Orig)':>8} {'H(A)':>8} {'H(B)':>8} {'|Delta|':>8}")
for v in COLS:
    da = abs(E_A[v] - E_B[v])
    print(f"{v:<10} {E_ORIG[v]:8.4f} {E_A[v]:8.4f} {E_B[v]:8.4f} {da:8.4f}")

print(f"\n  COMPARATIVA MST")
for i in range(max(len(ARISTAS_A), len(ARISTAS_B))):
    a_s = f"{ARISTAS_A[i][0]}--{ARISTAS_A[i][1]} ({ARISTAS_A[i][2]:.4f})" if i < len(ARISTAS_A) else "---"
    b_s = f"{ARISTAS_B[i][0]}--{ARISTAS_B[i][1]} ({ARISTAS_B[i][2]:.4f})" if i < len(ARISTAS_B) else "---"
    print(f"  {i+1:2d}  {a_s:<30}  {b_s:<30}")
print(f"\n  Peso MST Original: {PESO_ORIG:.4f}")
print(f"  Peso MST A: {PESO_A:.4f}")
print(f"  Peso MST B: {PESO_B:.4f}")

# ==============================
# 6. rVP
# ==============================
print(f"\n{'='*70}")
print(f"  ANALISIS rVP - Distancias de Camino")
print(f"{'='*70}")

def mst_a_grafo(aristas):
    G = nx.Graph()
    G.add_weighted_edges_from(aristas)
    return G

def matriz_distancias_mst(G, cols):
    n = len(cols)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                D[i][j] = 0
            else:
                try:
                    D[i][j] = nx.shortest_path_length(G, source=cols[i], target=cols[j], weight='weight')
                except:
                    D[i][j] = 0
    return pd.DataFrame(D, index=cols, columns=cols)

G_A = mst_a_grafo(ARISTAS_A)
G_B = mst_a_grafo(ARISTAS_B)

dist_A = matriz_distancias_mst(G_A, COLS)
dist_B = matriz_distancias_mst(G_B, COLS)

sum_A = dist_A.sum(axis=1)
sum_B = dist_B.sum(axis=1)
delta_rvp = sum_A - sum_B
abs_delta = delta_rvp.abs()

print(f"\n{'Variable':<10} {'Sum_A':>8} {'Sum_B':>8} {'Delta':>8} {'|Delta|':>8} {'Critica?':>10}")
for v in COLS:
    crit = "SI" if abs_delta[v] > 0.5 else "No"
    print(f"{v:<10} {sum_A[v]:8.4f} {sum_B[v]:8.4f} {delta_rvp[v]:8.4f} {abs_delta[v]:8.4f} {crit:>10}")

n_criticas = (abs_delta > 0.5).sum()
print(f"\n>>> Variables criticas (|Delta| > 0.5): {n_criticas} de {len(COLS)}")

# ==============================
# 7. RED BAYESIANA (Chow-Liu)
# ==============================
print(f"\n{'='*70}")
print(f"  RED BAYESIANA (Chow-Liu)")
print(f"{'='*70}")

print(f"\n>>> El MST construido ES la estructura de Chow-Liu.")
print(f">>> Chow-Liu: aprender estructura de red Bayesiana tipo arbol")
print(f">>> encontrando el arbol de expansion maxima sobre MI.")

# Usar el MST del dataset completo como estructura
print(f"\n>>> Estructura aprendida (Dataset Original):")
G_cl = nx.Graph()
G_cl.add_weighted_edges_from(ARISTAS_ORIG)

# Orientar desde Target como raiz usando BFS
dag = nx.DiGraph()
visited = set()
queue = ['Target']
visited.add('Target')
while queue:
    u = queue.pop(0)
    for v in G_cl.neighbors(u):
        if v not in visited:
            dag.add_edge(u, v, weight=G_cl[u][v]['weight'])
            visited.add(v)
            queue.append(v)

print("\n>>> Aristas dirigidas del DAG (Target como raiz):")
for u, v, data in dag.edges(data=True):
    print(f"    {u} -> {v}  (peso MI: {data['weight']:.4f})")

# Visualizar DAG
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(dag, seed=42, k=2.5, iterations=200)
nx.draw_networkx_nodes(dag, pos, node_size=2800, node_color='#FFB347', edgecolors='black', linewidths=2)
nx.draw_networkx_labels(dag, pos, font_size=10, font_weight='bold')
nx.draw_networkx_edges(dag, pos, width=2.5, edge_color='#555555',
                       arrows=True, arrowsize=30, arrowstyle='-|>',
                       node_size=2800)
labels = {(u, v): f'{data["weight"]:.4f}' for u, v, data in dag.edges(data=True)}
nx.draw_networkx_edge_labels(dag, pos, edge_labels=labels, font_size=8, font_color='red')
plt.title('Red Bayesiana - Chow-Liu\nStudent Dropout (Target como raiz)', fontsize=16, fontweight='bold', pad=20)
plt.axis('off')
plt.tight_layout()
plt.savefig('red_bayesiana.png', format='PNG', dpi=300, bbox_inches='tight')
plt.close()
print(">>> red_bayesiana.png")

# CPTs: P(Target | padre directo)
print(f"\n>>> Tablas de Probabilidad Condicional (CPTs):")
for u, v in dag.edges():
    if u == 'Target':
        # P(v | Target)
        cpt = df_orig.groupby(['Target', v]).size().unstack(fill_value=0)
        cpt = cpt.div(cpt.sum(axis=1), axis=0)
        print(f"\n  P({v} | Target):")
        print(cpt.round(4).to_string())

# Ejemplo de inferencia
print(f"\n>>> Ejemplo de inferencia Bayesiana:")
print(f"  Pregunta: P(Target | x1=0, x3=0)")
subset = df_orig[(df_orig['x1'] == 0) & (df_orig['x3'] == 0)]
if len(subset) > 0:
    probs = subset['Target'].value_counts(normalize=True).sort_index()
    for t, p in probs.items():
        label = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}[t]
        print(f"    P(Target={label} | x1=bajo, x3=no pago) = {p:.4f} ({p*100:.1f}%)")
else:
    print(f"    Sin datos para esa combinacion")

# ==============================
# 8. EXPORTAR
# ==============================
MI_A.to_csv('matriz_MI_A.csv', float_format='%.4f')
MI_B.to_csv('matriz_MI_B.csv', float_format='%.4f')
MI_ORIG.to_csv('matriz_MI_Original.csv', float_format='%.4f')

print(f"\n{'='*70}")
print(f"  ANALISIS COMPLETADO")
print(f"{'='*70}")
