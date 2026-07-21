import pandas as pd, numpy as np, networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from math import log
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("  3 ENFOQUES DE REDES BAYESIANAS (DAG)")
print("  Chow-Liu | Naive Bayes | Hill Climbing")
print("=" * 70)

df = pd.read_csv('data.csv', sep=';')
df.columns = df.columns.str.strip()
VARS = {
    'Admission grade': 'x1', 'Age at enrollment': 'x2',
    'Tuition fees up to date': 'x3', 'Scholarship holder': 'x4',
    'Curricular units 1st sem (approved)': 'x5',
    'Curricular units 1st sem (grade)': 'x6',
    'Curricular units 2nd sem (approved)': 'x7',
    'Curricular units 2nd sem (grade)': 'x8',
    'Marital status': 'x9',
}
df_w = df[list(VARS.keys()) + ['Target', 'Displaced']].copy()
df_w.rename(columns=VARS, inplace=True)
df_w['Target'] = df_w['Target'].map({'Dropout': 0, 'Enrolled': 1, 'Graduate': 2})
for c in ['x1', 'x2', 'x5', 'x6', 'x7', 'x8']:
    df_w[c] = pd.qcut(df_w[c], q=3, labels=[0, 1, 2], duplicates='drop')
df_w['x9'] = df_w['x9'].map(lambda x: 0 if x == 1 else 1)

df_orig = df_w.drop(columns=['Displaced'])
COLS = list(df_orig.columns)
n = len(df_orig)

from pgmpy.models import BayesianNetwork as BN
from pgmpy.estimators import HillClimbSearch

# Try newer API first, fall back
try:
    from pgmpy.models import DiscreteBayesianNetwork
    BayesianNetwork = DiscreteBayesianNetwork
except ImportError:
    BayesianNetwork = BN

def bic_score(edges, data):
    model = BayesianNetwork(edges)
    model.fit(data)
    log_lik = 0
    for node in nodes:
        parents = [p for p, c in edges if c == node]
        if parents:
            cpd = model.get_cpds(node)
            for _, row in data.iterrows():
                vals = tuple(row[[node] + parents])
                prob = cpd.get_value(**{node: vals[0]}, **dict(zip(parents, vals[1:])))
                if prob and prob > 0:
                    log_lik += log(prob)
        else:
            vals = data[node].value_counts(normalize=True)
            for v, p in vals.items():
                if p > 0:
                    log_lik += (data[node] == v).sum() * log(p)
    k = 0
    for node in nodes:
        parents = [p for p, c in edges if c == node]
        nv = len(data[node].unique())
        npv = 1
        for p in parents:
            npv *= len(data[p].unique())
        k += (nv - 1) * npv
    return -2 * log_lik + k * log(n), log_lik, k

nodes = sorted(COLS)

# =============================================
# 1. CHOW-LIU
# =============================================
print("\n>>> Enfoque 1: Chow-Liu (MST sobre MI)")

mi = pd.DataFrame(np.zeros((len(COLS), len(COLS))), index=COLS, columns=COLS)
for i in range(len(COLS)):
    for j in range(i, len(COLS)):
        if i != j:
            v = mutual_info_score(df_orig.iloc[:, i], df_orig.iloc[:, j]) / np.log(2)
            mi.iloc[i, j] = v; mi.iloc[j, i] = v

G_mi = nx.Graph()
for i in range(len(COLS)):
    for j in range(i + 1, len(COLS)):
        G_mi.add_edge(COLS[i], COLS[j], weight=mi.iloc[i, j])
G_neg = nx.Graph()
for u, v, d in G_mi.edges(data=True):
    G_neg.add_edge(u, v, weight=-d['weight'])
mst = nx.minimum_spanning_tree(G_neg)

dag_cl = nx.DiGraph()
visited = {'Target'}; queue = ['Target']
while queue:
    u = queue.pop(0)
    for v in nx.Graph(mst.edges()).neighbors(u):
        if v not in visited:
            dag_cl.add_edge(u, v, weight=G_mi[u][v]['weight'])
            visited.add(v); queue.append(v)

edges_cl = [(u, v) for u, v in dag_cl.edges()]
bic_cl, ll_cl, k_cl = bic_score(edges_cl, df_orig)
print(f"  Aristas: {len(edges_cl)}, BIC: {bic_cl:.2f}, params: {k_cl}")

# =============================================
# 2. NAIVE BAYES
# =============================================
print("\n>>> Enfoque 2: Naive Bayes")

edges_nb = [('Target', c) for c in ['x1','x2','x3','x4','x5','x6','x7','x8','x9']]
dag_nb = nx.DiGraph()
dag_nb.add_edges_from(edges_nb)
bic_nb, ll_nb, k_nb = bic_score(edges_nb, df_orig)
print(f"  Aristas: {len(edges_nb)}, BIC: {bic_nb:.2f}, params: {k_nb}")

# =============================================
# 3. HILL CLIMBING
# =============================================
print("\n>>> Enfoque 3: Hill Climbing")

hc = HillClimbSearch(df_orig)
best_model = hc.estimate()
edges_hc = list(best_model.edges())
dag_hc = nx.DiGraph()
dag_hc.add_edges_from(edges_hc)
bic_hc, ll_hc, k_hc = bic_score(edges_hc, df_orig)
print(f"  Aristas: {len(edges_hc)}, BIC: {bic_hc:.2f}, params: {k_hc}")
print(f"  Aristas: {edges_hc}")

# =============================================
# 4. COMPARATIVA
# =============================================
print(f"\n{'='*70}")
print(f"  COMPARATIVA DE DAGs")
print(f"{'='*70}")
print(f"  {'Modelo':<20} {'Aristas':>8} {'BIC':>12} {'Params':>8}")
print(f"  {'-'*48}")
print(f"  {'Chow-Liu':<20} {len(edges_cl):>8} {bic_cl:>12.2f} {k_cl:>8}")
print(f"  {'Naive Bayes':<20} {len(edges_nb):>8} {bic_nb:>12.2f} {k_nb:>8}")
print(f"  {'Hill Climbing':<20} {len(edges_hc):>8} {bic_hc:>12.2f} {k_hc:>8}")

scores = [('Chow-Liu', bic_cl, len(edges_cl)),
          ('Naive Bayes', bic_nb, len(edges_nb)),
          ('Hill Climbing', bic_hc, len(edges_hc))]
mejor = min(scores, key=lambda x: x[1])
print(f"\n  Mejor modelo (menor BIC): {mejor[0]} (BIC={mejor[1]:.2f})")

# =============================================
# 5. GRAFICAR
# =============================================
dags_info = [
    ('chow_liu', 'Chow-Liu (MST sobre MI)', dag_cl, '#87CEFA'),
    ('naive_bayes', 'Naive Bayes (Target -> features)', dag_nb, '#90EE90'),
    ('hill_climbing', 'Hill Climbing (Score BIC)', dag_hc, '#FFB347'),
]

for key, titulo, dag, color in dags_info:
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(dag, seed=42, k=2.5, iterations=200)
    nx.draw_networkx_nodes(dag, pos, node_size=2800, node_color=color,
                           edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(dag, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(dag, pos, width=2.5, edge_color='#555555',
                           arrows=True, arrowsize=30, arrowstyle='-|>',
                           node_size=2800)
    if key == 'chow_liu':
        labels = {(u, v): f'{G_mi[u][v]["weight"]:.4f}' for u, v in dag.edges()}
        nx.draw_networkx_edge_labels(dag, pos, edge_labels=labels,
                                     font_size=8, font_color='red')
    plt.title(f'Red Bayesiana - {titulo}\nStudent Dropout', fontsize=16,
              fontweight='bold', pad=20)
    plt.axis('off'); plt.tight_layout()
    nombre = f'dag_{key}.png'
    plt.savefig(nombre, format='PNG', dpi=300, bbox_inches='tight'); plt.close()
    print(f">>> {nombre}")

# =============================================
# 6. EXPORTAR COMPARATIVA
# =============================================
with open('tabla_comparativa_dags.tex', 'w', encoding='utf-8') as f:
    f.write('\\begin{table}[H]\n\\centering\n')
    f.write('\\begin{tabular}{lcccc}\n\\toprule\n')
    f.write('\\textbf{Modelo} & \\textbf{Aristas} & \\textbf{BIC} & \\textbf{Log-Lik} & \\textbf{Params} \\\\\n\\midrule\n')
    f.write(f'Chow-Liu & {len(edges_cl)} & {bic_cl:.2f} & {ll_cl:.2f} & {k_cl} \\\\\n')
    f.write(f'Naive Bayes & {len(edges_nb)} & {bic_nb:.2f} & {ll_nb:.2f} & {k_nb} \\\\\n')
    f.write(f'Hill Climbing & {len(edges_hc)} & {bic_hc:.2f} & {ll_hc:.2f} & {k_hc} \\\\\n')
    f.write('\\midrule\n')
    f.write(f'\\rowcolor{{blue!15}}\n')
    f.write(f'\\textbf{{Mejor}} ({mejor[0]}) & & \\textbf{{{mejor[1]:.2f}}} & & \\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n')
    f.write('\\caption{Comparativa de los tres enfoques de Redes Bayesianas. Menor BIC = mejor modelo.}\n\\end{table}\n')

print(f"\n>>> tabla_comparativa_dags.tex generado")
print(f"\n{'='*70}")
print(f"  3 DAGs COMPLETADO")
print(f"{'='*70}")
