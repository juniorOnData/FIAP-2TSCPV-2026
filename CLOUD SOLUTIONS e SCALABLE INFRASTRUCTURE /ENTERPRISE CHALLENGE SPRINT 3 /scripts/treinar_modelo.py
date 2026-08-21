"""
treinar_modelo.py
-----------------
Gera o arquivo modelo_incidentes.pkl que a API vai carregar dentro do container.

Se o time de Machine Learning já entregou um .pkl pronto, VOCÊ NÃO PRECISA DESTE SCRIPT:
basta colocar o arquivo modelo_incidentes.pkl na mesma pasta do Dockerfile.

Este script existe para você não ficar bloqueado esperando a outra disciplina.
Ele treina uma regressão linear simples que prevê a quantidade de incidentes de
amanhã (D+1) a partir dos últimos dias.

Como rodar (no seu computador ou no Cloud Shell):
    pip install pandas scikit-learn joblib
    python treinar_modelo.py incidentes.csv

O CSV de entrada precisa ter, no mínimo, uma coluna de data e uma de contagem.
Ajuste os nomes em COLUNA_DATA e COLUNA_QTD abaixo.
"""

import sys
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

COLUNA_DATA = "data"
COLUNA_QTD = "qtd_incidentes"


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Cria as variáveis (features) que o modelo usa para prever."""
    df = df.copy()
    df[COLUNA_DATA] = pd.to_datetime(df[COLUNA_DATA])
    df = df.sort_values(COLUNA_DATA).reset_index(drop=True)

    # Lags: quantos incidentes houve 1, 2 e 7 dias atrás
    df["lag_1"] = df[COLUNA_QTD].shift(1)
    df["lag_2"] = df[COLUNA_QTD].shift(2)
    df["lag_7"] = df[COLUNA_QTD].shift(7)
    # Média móvel dos últimos 7 dias (tendência recente)
    df["media_7"] = df[COLUNA_QTD].shift(1).rolling(7).mean()
    # Dia da semana (0 = segunda) e se é fim de semana
    df["dia_semana"] = df[COLUNA_DATA].dt.dayofweek
    df["fim_de_semana"] = (df["dia_semana"] >= 5).astype(int)

    return df.dropna().reset_index(drop=True)


def main(caminho_csv: str) -> None:
    df = pd.read_csv(caminho_csv)

    # Se o CSV for a base bruta de chamados (1 linha por incidente),
    # agregamos por dia para virar uma série temporal.
    if COLUNA_QTD not in df.columns:
        df[COLUNA_DATA] = pd.to_datetime(df[COLUNA_DATA])
        df = (
            df.groupby(df[COLUNA_DATA].dt.date)
            .size()
            .reset_index(name=COLUNA_QTD)
            .rename(columns={df.columns[0]: COLUNA_DATA})
        )

    base = preparar(df)
    features = ["lag_1", "lag_2", "lag_7", "media_7", "dia_semana", "fim_de_semana"]

    X = base[features]
    y = base[COLUNA_QTD]

    # Separação temporal: 80% mais antigos para treino, 20% mais recentes para teste.
    # NUNCA embaralhe séries temporais — isso causa data leakage.
    corte = int(len(base) * 0.8)
    modelo = LinearRegression().fit(X[:corte], y[:corte])

    r2 = modelo.score(X[corte:], y[corte:])
    print(f"Linhas usadas: {len(base)} | R2 no teste: {r2:.3f}")

    # Salvamos o modelo E a lista de features, para a API montar a entrada na ordem certa.
    joblib.dump({"modelo": modelo, "features": features}, "modelo_incidentes.pkl")
    print("Arquivo modelo_incidentes.pkl gerado com sucesso.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "incidentes.csv")
