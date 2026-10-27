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
# 1. Função Otimizada para Cálculo do Divergente Hierárquico (BFS)
# =============================================================================
def calculate_divergence_bfs(G, max_H=8):
    """
    Calcula o divergente hierárquico para H de 1 a max_H usando BFS.
    Retorna um dicionário contendo 8 dicionários internos: {H: {no: valor_div}}
    """
    degrees = dict(G.degree())
    nodes = list(G.nodes())
    
    # Inicializa os 8 dicionários (H=1 a H=8)
    div_dicts = {H: {n: 0.0 for n in nodes} for H in range(1, max_H + 1)}

    for start_node in nodes:
        k_i = degrees[start_node]
        
        # BFS Setup
        visited = {start_node}
        queue = deque([(start_node, 0)]) # tupla: (nó, distancia)
        
        # Agrupa vizinhos pela distância exata
        H_neighbors = defaultdict(list)
        
        while queue:
            curr_node, dist = queue.popleft()
            
            if dist > 0:
                H_neighbors[dist].append(curr_node)
                
            # Só explora até max_H
            if dist < max_H:
                for neighbor in G.neighbors(curr_node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
        
        # Calcula a métrica para cada H
        for H in range(1, max_H + 1):
            neighbors_at_H = H_neighbors.get(H, [])
            if len(neighbors_at_H) > 0:
                div_val = sum(degrees[j] - k_i for j in neighbors_at_H) / len(neighbors_at_H)
                div_dicts[H][start_node] = div_val
            else:
                div_dicts[H][start_node] = 0.0 # Nó sem vizinhos na profundidade H
                
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
    Rodará isolado em 1 dos 8 núcleos alocados.
    """
    # Monta o dataset (X) concatenando as listas de v_H até H_max
    X_concat = []
    for net_feats in X_all_H:
        concatenated_vector = []
        for i in range(H_max):
            concatenated_vector.extend(net_feats[i])
        X_concat.append(concatenated_vector)
        
    X_concat = np.array(X_concat)
    
    svm = SVC(kernel='rbf', random_state=42)
    # n_jobs=1 aqui porque o paralelismo já está ocorrendo fora desta função (nos 8 níveis)
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
    "/home/DATA/datasets/hierarchical_divergence/networks/synthetic_model/graphs/"
]

output_csv = "/home/jvmerenda/hierarchical_divergence/H13/Hdiv_outputs_H13.csv"

# Cria/limpa o arquivo CSV e escreve o cabeçalho
with open(output_csv, mode='w', newline='') as file:
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
        
    X_all_H = []
    y_labels = []
    
    # 4.1 Loop pelas Redes
    for idx, filepath in enumerate(files, 1):
        filename = os.path.basename(filepath)
        label = filename.split('_')[0]
        y_labels.append(label)
        
        print(f"\r  -> Extraindo rede {idx} de {total_files} | Classe: {label}", end="", flush=True)
        
        # Carrega lista de adjacência (usamos nodetype=int para eficiência e consistência)
        G = nx.read_adjlist(filepath, nodetype=int)
        
        div_dicts = calculate_divergence_bfs(G, max_H=13)
        
        network_features = []
        for H in range(1, 14):
            v_H = extract_vH(div_dicts[H])
            network_features.append(v_H)
            
        X_all_H.append(network_features)
        
    print("\n  -> Extração concluída. Iniciando Classificação Paralela em 8 núcleos...")
    
    # Codifica rótulos (strings para ints)
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    
    # Trata caso do protist (k=5) vs outros (k=10)
    k_folds = 5 if 'protist' in dataset_path.lower() else 10
    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # 4.2 Paralelismo: Dispara os 8 níveis de concatenação simultaneamente
    # joblib.Parallel rodará cada evaluate_concatenation_level em um núcleo diferente
    resultados_niveis = Parallel(n_jobs=8)(
        delayed(evaluate_concatenation_level)(
            H_max, 
            X_all_H, 
            y, 
            cv, 
            "[" + ", ".join([f"H = {i}" for i in range(1, H_max + 1)]) + "]"
        ) for H_max in range(1, 14)
    )
    
    # 4.3 Salva os resultados processados em paralelo no CSV (sequencialmente para evitar conflito de I/O)
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        
        # 'resultados_niveis' é uma lista com as tuplas retornadas pelos 8 processos paralelos
        for concat_label, mean_acc, std_acc in resultados_niveis:
            print(f"     Nível: {concat_label: <40} | Acc: {mean_acc:.4f} ± {std_acc:.4f}")
            writer.writerow([dataset_path, concat_label, round(mean_acc, 6), round(std_acc, 6)])

print(f"\nExperimento finalizado com sucesso! Resultados salvos em: {output_csv}")