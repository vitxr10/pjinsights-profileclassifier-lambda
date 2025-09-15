import json
import numpy as np
import os

# === Configurações ===
MIN_SALDO_TRAIN = -85900900.0  

CLUSTER_TO_PROFILE = {
    0: "expansao",
    1: "inicio",
    2: "declinio",
    3: "maturidade"
}

# === Carregar parâmetros do scaler e do KMeans ===
base_path = os.path.join(os.path.dirname(__file__), "params")

with open(os.path.join(base_path, "scaler_params.json"), "r") as f:
    scaler_params = json.load(f)

with open(os.path.join(base_path, "kmeans_params.json"), "r") as f:
    kmeans_params = json.load(f)

# Transforma listas em arrays numpy
scaler_mean_ = np.array(scaler_params["mean"])
scaler_scale_ = np.array(scaler_params["scale"])
kmeans_cluster_centers_ = np.array(kmeans_params["cluster_centers"])

# === Funções utilitárias ===
def preprocess_empresa(empresa: dict):
    """Recebe uma empresa (dict) e gera os features transformados."""
    vl_fatu = empresa["VL_FATU"]
    vl_sldo = empresa["VL_SLDO"]

    vl_fatu_log = np.log1p(vl_fatu)
    vl_sldo_shifted = vl_sldo - MIN_SALDO_TRAIN + 1
    vl_sldo_log = np.log(vl_sldo_shifted)

    print(f"[LOG] Pré-processado: {empresa['ID']} -> VL_FATU_log: {vl_fatu_log:.2f}, VL_SLDO_log: {vl_sldo_log:.2f}")
    return np.array([vl_fatu_log, vl_sldo_log])

def scale_features(features: np.ndarray):
    """Aplica a mesma transformação do StandardScaler."""
    scaled = (features - scaler_mean_) / scaler_scale_
    print(f"[LOG] Features escalados:\n{scaled}")
    return scaled

def predict_kmeans(features_scaled: np.ndarray):
    """Predição manual do cluster mais próximo (distância euclidiana)."""
    distances = np.linalg.norm(
        features_scaled[:, np.newaxis, :] - kmeans_cluster_centers_,
        axis=2
    )
    clusters = np.argmin(distances, axis=1)
    print(f"[LOG] Clusters previstos: {clusters}")
    return clusters

# === Lambda Handler ===
def lambda_handler(event, context):
    print('[LOG] Iniciando lambda ProfileClassifier...')
    print(f"[LOG] Evento recebido: {event}")

    try:
        body = event["body"]

        # Se body for string (API Gateway), faz o json.loads
        if isinstance(body, str):
            body = json.loads(body)

        # Garantir que body é uma lista
        if not isinstance(body, list):
            body = [body]

        print(f"[LOG] Corpo da requisição processado: {body}")

        # Pré-processa
        features = np.array([preprocess_empresa(emp) for emp in body])
        features_scaled = scale_features(features)
        clusters = predict_kmeans(features_scaled)

        # Montar resposta
        response = []
        for emp, cluster in zip(body, clusters):
            emp_rotulado = emp.copy()
            emp_rotulado["PERFIL"] = CLUSTER_TO_PROFILE[int(cluster)]
            response.append(emp_rotulado)

        print(f"[LOG] Corpo da resposta: {response}")

        return {
            "statusCode": 200,
            "body": json.dumps(response, ensure_ascii=False)
        }

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
