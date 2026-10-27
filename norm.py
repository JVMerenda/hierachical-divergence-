import os
import glob
import numpy as np
import networkx as nx
from scipy.stats import entropy
from collections import deque, defaultdict
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
import csv

# =============================================================================
# 1. Função Otimizada para Cálculo das Métricas de Divergente Hierárquico (BFS)
# =============================================================================
def calculate_divergence_bfs_all_metrics(G, max_H=13):
    """
    Calcula as 4 métricas alternativas de divergente hierárquico para H de 1 a max_H usando BFS:
      1. sum: div^Sigma(i, H) = sum_{j in Omega} (k_j - k_i)
      2. median: div^med(i, H) = median_{j in Omega}(k_j) - k_i
      3. weighted: div^w(i, H) = sum_{j in Omega} w_{ij} * (k_j - k_i), onde w_{ij} = sigma_{ij} / sum_{l in Omega} sigma_{il}
      4. standardized: div^z(i, H) = (mu_Omega - k_i) / s_Omega (se s_Omega > 0, senão 0)
      
    Retorna um dicionário:
      {
        'sum': {H: {no: valor}},
        'median': {H: {no: valor}},
        'weighted': {H: {no: valor}},
        'standardized': {H: {no: valor}}
      }
    """
    degrees = dict(G.degree())
    nodes = list(G.nodes())
    metrics = ['sum', 'median', 'weighted', 'standardized']
    
    # Inicializa os dicionários para cada métrica e nível H (1 a max_H)
    div_dicts = {
        m: {H: {n: 0.0 for n in nodes} for H in range(1, max_H + 1)}
        for m in metrics
    }

    for start_node in nodes:
        k_i = degrees[start_node]
        
        # BFS Setup com contagem de caminhos mínimos (sigma)
        dist = {start_node: 0}
        sigma = {start_node: 1}  # sigma[v] guarda a quantidade de caminhos mínimos de start_node a v
        queue = deque([start_node])
        
        # Agrupa vizinhos pela distância exata H
        H_neighbors = defaultdict(list)
        
        while queue:
            curr_node = queue.popleft()
            d_curr = dist[curr_node]
            
            if d_curr > 0:
                H_neighbors[d_curr].append(curr_node)
                
            # Explora vizinhos até max_H
            if d_curr < max_H:
                for neighbor in G.neighbors(curr_node):
                    if neighbor not in dist:
                        dist[neighbor] = d_curr + 1
                        sigma[neighbor] = sigma[curr_node]
                        queue.append(neighbor)
                    elif dist[neighbor] == d_curr + 1:
                        sigma[neighbor] += sigma[curr_node]
        
        # Calcula as 4 métricas para cada H
        for H in range(1, max_H + 1):
            neighbors_at_H = H_neighbors.get(H, [])
            if len(neighbors_at_H) > 0:
                deg_diffs = [degrees[j] - k_i for j in neighbors_at_H]
                deg_vals = [degrees[j] for j in neighbors_at_H]
                
                # 1. Sum (soma direta sem divisão por s)
                div_dicts['sum'][H][start_node] = float(np.sum(deg_diffs))
                
                # 2. Median (mediana dos graus da casca - k_i)
                div_dicts['median'][H][start_node] = float(np.median(deg_vals) - k_i)
                
                # 3. Weighted (média ponderada pelo número de caminhos mínimos sigma_ij)
                sigmas_at_H = [sigma[j] for j in neighbors_at_H]
                total_sigma = sum(sigmas_at_H)
                if total_sigma > 0:
                    weighted_div = sum(sigmas_at_H[idx] * deg_diffs[idx] for idx in range(len(neighbors_at_H))) / total_sigma
                else:
                    weighted_div = 0.0
                div_dicts['weighted'][H][start_node] = float(weighted_div)
                
                # 4. Standardized (div. padronizada pelo desvio padrão dos graus da casca)
                s_omega = np.std(deg_vals)
                if s_omega > 0:
                    mu_omega = np.mean(deg_vals)
                    z_div = (mu_omega - k_i) / s_omega
                else:
                    z_div = 0.0
                div_dicts['standardized'][H][start_node] = float(z_div)
            else:
                # Convenção para casca vazia (Omega = Ø)
                div_dicts['sum'][H][start_node] = 0.0
                div_dicts['median'][H][start_node] = 0.0
                div_dicts['weighted'][H][start_node] = 0.0
                div_dicts['standardized'][H][start_node] = 0.0
                
    return div_dicts

# =============================================================================
# 2. Extração do Feature Vector (v_H)
# =============================================================================
def extract_vH(div_dict_H):
    """
    Extrai: [média, desvio_padrão, entropia, Q1, Q2(mediana), Q3, Q4(max)]
    """
    vals = list(div_dict_H.values())
    
    if len(vals) == 0:
        return [0.0] * 7
        
    mean_val = np.mean(vals)
    std_val = np.std(vals)
    
    # Cálculo da entropia aproximada usando histograma
    counts, _ = np.histogram(vals, bins=10, density=False)
    probs = counts / np.sum(counts) if np.sum(counts) > 0 else counts
    ent = entropy(probs)
    
    # 4 Quartis (25%, 50%, 75%, 100%)
    q1, q2, q3, q4 = np.percentile(vals, [25, 50, 75, 100])
    
    return [mean_val, std_val, ent, q1, q2, q3, q4]

# =============================================================================
# 3. Função Auxiliar para Paralelismo (Executa uma Classificação de Nível)
# =============================================================================
def evaluate_concatenation_level(H_max, X_all_H, y, cv, concat_label):
    """
    Monta as features até H_max e roda o SVM com cross-validation.
    """
    X_concat = []
    for net_feats in X_all_H:
        concatenated_vector = []
        for i in range(H_max):
            concatenated_vector.extend(net_feats[i])
        X_concat.append(concatenated_vector)
        
    X_concat = np.array(X_concat)
    
    svm = SVC(kernel='rbf', random_state=42)
    scores = cross_val_score(svm, X_concat, y, cv=cv, scoring='accuracy', n_jobs=1)
    
    return concat_label, np.mean(scores), np.std(scores)

# =============================================================================
# 4. Pipeline Principal
# =============================================================================
datasets = [
    "/home/jvmerenda/hierarchical_divergence/networks/actinobacteria/graphs/",
    "/home/jvmerenda/hierarchical_divergence/networks/animals/graphs/",
    "/home/jvmerenda/hierarchical_divergence/networks/firmicutes_bacillis/graphs/",
    "/home/jvmerenda/hierarchical_divergence/networks/fungi/graphs/",
    "/home/jvmerenda/hierarchical_divergence/networks/kingdom/graphs/",
    "/home/jvmerenda/hierarchical_divergence/networks/plant/graphs/",
    "/home/jvmerenda/hierarchical_divergence/networks/protist/graphs/",
    "/home/jvmerenda/hierarchical_divergence/networks/ENZYMES/",
    "/home/jvmerenda/hierarchical_divergence/networks/PROTEINS/",
    "/home/jvmerenda/hierarchical_divergence/networks/MUTAG/",
    "/home/jvmerenda/hierarchical_divergence/networks/IMDB-MULTI/",
    "/home/jvmerenda/hierarchical_divergence/networks/COLLAB/",
    "/home/jvmerenda/hierarchical_divergence/networks/REDDIT-BINARY/",
    "/home/jvmerenda/hierarchical_divergence/networks/REDDIT-MULTI-5K/",
    "/home/hierarchical_divergence/networks/synthetic_model/graphs/"
]

output_dir = "/home/jvmerenda/hierarchical_divergence/other_normalizations/"
os.makedirs(output_dir, exist_ok=True)

metrics = ["sum", "median", "weighted", "standardized"]
output_csvs = {
    m: os.path.join(output_dir, f"Hdiv_outputs_{m}.csv")
    for m in metrics
}

# Inicializa os 4 arquivos CSV e escreve os cabeçalhos
for metric_name, csv_path in output_csvs.items():
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["dataset", "nivel_de_concatenacao", "mean_accuracy", "standard_deviation"])

for dataset_path in datasets:
    print(f"\n{'='*60}")
    print(f"PROCESSANDO DATASET: {dataset_path}")
    print(f"{'='*60}")
    
    files = glob.glob(os.path.join(dataset_path, "*.txt"))
    total_files = len(files)
    
    if total_files == 0:
        print("Nenhum arquivo .txt encontrado. Pulando dataset.")
        continue
        
    # Armazena as características extraídas para cada métrica
    X_all_H_by_metric = {m: [] for m in metrics}
    y_labels = []
    
    # 4.1 Loop pelas Redes
    for idx, filepath in enumerate(files, 1):
        filename = os.path.basename(filepath)
        label = filename.split('_')[0]
        y_labels.append(label)
        
        print(f"\r  -> Extraindo rede {idx} de {total_files} | Classe: {label}", end="", flush=True)
        
        G = nx.read_adjlist(filepath, nodetype=int)
        
        # Calcula as 4 métricas no mesmo ciclo BFS
        div_dicts = calculate_divergence_bfs_all_metrics(G, max_H=13)
        
        for m in metrics:
            network_features_m = []
            for H in range(1, 14):
                v_H = extract_vH(div_dicts[m][H])
                network_features_m.append(v_H)
            X_all_H_by_metric[m].append(network_features_m)
            
    print("\n  -> Extração concluída.")
    
    # Codifica rótulos (strings para ints)
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    
    k_folds = 5 if 'protist' in dataset_path.lower() else 10
    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # 4.2 Avaliação paralela e gravação dos resultados para cada métrica
    for m in metrics:
        print(f"  [Métrica: {m.upper()}] Executando classificação paralela nos 8 núcleos...")
        X_all_H = X_all_H_by_metric[m]
        
        resultados_niveis = Parallel(n_jobs=8)(
            delayed(evaluate_concatenation_level)(
                H_max, 
                X_all_H, 
                y, 
                cv, 
                "[" + ", ".join([f"H = {i}" for i in range(1, H_max + 1)]) + "]"
            ) for H_max in range(1, 14)
        )
        
        # Escreve os resultados no CSV correspondente à métrica
        with open(output_csvs[m], mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            for concat_label, mean_acc, std_acc in resultados_niveis:
                print(f"     [{m}] Nível: {concat_label: <40} | Acc: {mean_acc:.4f} ± {std_acc:.4f}")
                writer.writerow([dataset_path, concat_label, round(mean_acc, 6), round(std_acc, 6)])

print(f"\nExperimento finalizado com sucesso!")
print(f"Resultados salvos no diretório: {output_dir}")
for m, csv_p in output_csvs.items():
    print(f"  - CSV Métrica '{m}': {csv_p}")