import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import mlflow
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODEL_URI = "models:/fiap_analytics.ml.previsao_kpi_gbm@champion"

# Threshold definido no seu modelo/notebook
ALERT_THRESHOLD = float(
    os.getenv("ALERT_THRESHOLD", "0.8958")
)

model = None


# ============================================================
# FEATURES EXATAMENTE COMO O MODELO ESPERA
# ============================================================

MODEL_FEATURES = [
    "tipo_alerta",
    "origem_abertura",
    "produto",
    "item_configuracao",
    "prioridade_origem",
    "dia_semana_abertura",
    "prioridade_cod",
    "hora_abertura",
    "fl_tem_incidente_pai",
    "fl_automatico",
]


# ============================================================
# SCHEMA DE ENTRADA
# ============================================================

class PredictionRequest(BaseModel):

    # Metadado do evento.
    # Não vai para o modelo.
    numero: str = Field(
        ...,
        description="Número do incidente"
    )

    # ========================================================
    # FEATURES DO MODELO
    # ========================================================

    tipo_alerta: Optional[str] = None

    origem_abertura: Optional[str] = None

    produto: Optional[str] = None

    item_configuracao: Optional[str] = None

    prioridade_origem: Optional[str] = None

    dia_semana_abertura: Optional[str] = None

    prioridade_cod: Optional[float] = None

    hora_abertura: Optional[float] = None

    fl_tem_incidente_pai: Optional[bool] = None

    fl_automatico: Optional[bool] = None


# ============================================================
# SCHEMA DE SAÍDA
# ============================================================

class PredictionResponse(BaseModel):

    numero: str

    probabilidade_nao_violacao: float

    probabilidade_violacao: float

    threshold_alerta: float

    alerta: bool

    model_uri: str

    prediction_timestamp: str


# ============================================================
# CARREGAMENTO DO MODELO
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global model

    print("=" * 70)
    print("CARREGANDO MODELO DE PREVISAO DE VIOLACAO")
    print("=" * 70)

    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")

    if not host:
        raise RuntimeError(
            "DATABRICKS_HOST nao configurado."
        )

    if not token:
        raise RuntimeError(
            "DATABRICKS_TOKEN nao configurado."
        )

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    print(f"Model URI: {MODEL_URI}")
    print()

    model = mlflow.pyfunc.load_model(
        MODEL_URI
    )

    print("=" * 70)
    print("MODELO CARREGADO COM SUCESSO")
    print("=" * 70)
    print()

    yield

    print("Encerrando model-service...")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="KPI Violation Prediction Service",
    description=(
        "Serviço de inferência em tempo real para previsão "
        "de violação de KPI."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_uri": MODEL_URI,
        "threshold": ALERT_THRESHOLD,
    }


# ============================================================
# PREDICT
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(payload: PredictionRequest):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo ainda nao carregado."
        )

    try:

        # ====================================================
        # TRANSFORMA REQUEST EM DICT
        # ====================================================

        payload_dict = payload.model_dump()

        numero = payload_dict.pop("numero")

        # ====================================================
        # GARANTE QUE APENAS FEATURES DO MODELO SEJAM USADAS
        # ====================================================

        features = {
            feature: payload_dict.get(feature)
            for feature in MODEL_FEATURES
        }

        # ====================================================
        # DATAFRAME
        # ====================================================

        df = pd.DataFrame(
            [features],
            columns=MODEL_FEATURES
        )

        print()
        print("=" * 70)
        print(f"PREDICAO - INCIDENTE {numero}")
        print("=" * 70)

        print(df.to_dict(orient="records")[0])

        # ====================================================
        # INFERÊNCIA
        #
        # O artefato MLflow foi registrado utilizando:
        #
        # pyfunc_predict_fn="predict_proba"
        #
        # portanto model.predict() devolve probabilidades.
        # ====================================================

        prediction = model.predict(df)

        probabilities = np.asarray(
            prediction
        )

        # Esperamos:
        #
        # [[P(classe 0), P(classe 1)]]
        #
        if (
            probabilities.ndim != 2
            or probabilities.shape[0] != 1
            or probabilities.shape[1] < 2
        ):

            raise RuntimeError(
                f"Formato inesperado retornado pelo modelo: "
                f"{probabilities.shape}"
            )

        prob_nao_violacao = float(
            probabilities[0][0]
        )

        prob_violacao = float(
            probabilities[0][1]
        )

        # ====================================================
        # REGRA DE ALERTA
        # ====================================================

        alerta = (
            prob_violacao >= ALERT_THRESHOLD
        )

        prediction_timestamp = (
            datetime.now(timezone.utc)
            .isoformat()
        )

        print(
            f"Probabilidade violacao: "
            f"{prob_violacao:.6f}"
        )

        print(
            f"Threshold: "
            f"{ALERT_THRESHOLD:.6f}"
        )

        print(
            f"Alerta: {alerta}"
        )

        # ====================================================
        # RESPONSE
        # ====================================================

        return PredictionResponse(

            numero=numero,

            probabilidade_nao_violacao=round(
                prob_nao_violacao,
                6
            ),

            probabilidade_violacao=round(
                prob_violacao,
                6
            ),

            threshold_alerta=ALERT_THRESHOLD,

            alerta=alerta,

            model_uri=MODEL_URI,

            prediction_timestamp=prediction_timestamp,
        )

    except Exception as e:

        print()
        print("ERRO DURANTE INFERENCIA:")
        print(repr(e))

        raise HTTPException(
            status_code=500,
            detail=f"Erro durante inferencia: {str(e)}"
        )
