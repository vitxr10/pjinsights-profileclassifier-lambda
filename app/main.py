import json
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Valor mínimo de saldo do treinamento (ajuste conforme seu dado real)
MIN_SALDO_TRAIN = -85900900.0

def load_pickle_models():
    """Carrega o modelo KMeans e o scaler dos arquivos pickle."""
    print("[LOG] Iniciando carregamento dos modelos pickle...")
    try:
        base_path = os.path.join(os.path.dirname(__file__), "pickle_files")
        print(f"[LOG] Caminho base dos modelos: {base_path}")
        with open(os.path.join(base_path, "kmeans_model_k4.pkl"), "rb") as f_model:
            kmeans_model = pickle.load(f_model)
        print("[LOG] Modelo KMeans carregado.")
        with open(os.path.join(base_path, "scaler.pkl"), "rb") as f_scaler:
            scaler = pickle.load(f_scaler)
        print("[LOG] Scaler carregado.")
        return kmeans_model, scaler
    except Exception as e:
        print(f"[ERRO] Falha ao carregar modelos: {e}")
        raise RuntimeError(f"Erro ao carregar modelos: {e}")

def preprocess_input(data_list):
    """Transforma a lista de objetos JSON em DataFrame e aplica as transformações necessárias."""
    print("[LOG] Iniciando preprocessamento dos dados de entrada...")
    try:
        df = pd.DataFrame(data_list)
        print(f"[LOG] DataFrame inicial: {df.shape}")
        df = df.rename(columns={
            "cnpj": "ID",
            "totalInvoicing": "VL_FATU",
            "totalBalance": "VL_SLDO",
            "openingDate": "DT_ABRT",
            "cnae": "DS_CNAE"
        })
        df["DT_ABRT"] = pd.to_datetime(df["DT_ABRT"])
        df_transformed = df.copy()
        df_transformed["VL_FATU_log"] = np.log1p(df_transformed["VL_FATU"])
        df_transformed["VL_SLDO_shifted"] = df_transformed["VL_SLDO"] - MIN_SALDO_TRAIN + 1
        df_transformed["VL_SLDO_log"] = np.log(df_transformed["VL_SLDO_shifted"])
        features = df_transformed[["VL_FATU_log", "VL_SLDO_log"]]
        print(f"[LOG] Features transformadas: {features.shape}")
        return df, features
    except Exception as e:
        print(f"[ERRO] Falha no preprocessamento: {e}")
        raise ValueError(f"Erro ao processar dados de entrada: {e}")

def predict_profiles(kmeans_model, scaler, features, df_original):
    """Aplica o scaler e o modelo para prever os clusters."""
    print("[LOG] Iniciando predição dos perfis...")
    try:
        features_scaled = scaler.transform(features)
        print(f"[LOG] Features escaladas: {features_scaled.shape}")
        predicted_clusters = kmeans_model.predict(features_scaled)
        print(f"[LOG] Clusters previstos: {predicted_clusters}")
        df_original["profile"] = predicted_clusters
        return df_original
    except Exception as e:
        print(f"[ERRO] Falha na predição: {e}")
        raise RuntimeError(f"Erro ao realizar predição: {e}")

def lambda_handler(event, context):
    print("[LOG] Lambda handler iniciado.")
    try:
        print(f"[LOG] Evento recebido: {event}")
        # Recebe o body da requisição
        body = event.get("body")
        if body is None:
            print("[ERRO] Body da requisição não encontrado.")
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Body da requisição não encontrado."})
            }
        # Se vier como string, converte para lista de dicts
        if isinstance(body, str):
            data_list = json.loads(body)
        else:
            data_list = body

        print(f"[LOG] Dados recebidos para classificação: {data_list}")

        # Carrega modelos
        kmeans_model, scaler = load_pickle_models()

        # Preprocessa dados
        df_original, features = preprocess_input(data_list)

        # Predição
        df_result = predict_profiles(kmeans_model, scaler, features, df_original)

        # Monta resposta mantendo campos originais + profile
        response_list = []
        for _, row in df_result.iterrows():
            response_list.append({
                "cnpj": row["ID"],
                "totalInvoicing": row["VL_FATU"],
                "totalBalance": row["VL_SLDO"],
                "openingDate": row["DT_ABRT"].strftime("%Y-%m-%d"),
                "cnae": row["DS_CNAE"],
                "profile": int(row["profile"])
            })

        print(f"[LOG] Resposta gerada: {response_list}")
        return {
            "statusCode": 200,
            "body": json.dumps(response_list)
        }

    except Exception as e:
        print(f"[ERRO] Exceção no handler: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }