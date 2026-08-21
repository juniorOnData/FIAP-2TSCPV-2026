import os
import logging
from datetime import date, timedelta
from io import StringIO

import joblib
import pandas as pd
import mysql.connector
from fastapi import FastAPI, HTTPException

try:
    from azure.monitor.opentelemetry import configure_azure_monitor

    if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
        configure_azure_monitor()
except Exception as exc:  # pragma: no cover
    print(f"[aviso] Application Insights nao configurado: {exc}")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("nexusops")

app = FastAPI(title="NexusOps - API de Previsao de Incidentes", version="1.0")

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "adminnexus")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "db_nexusops")
STORAGE_CONN = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
BLOB_CONTAINER = os.getenv("BLOB_CONTAINER", "raw")
BLOB_ARQUIVO = os.getenv("BLOB_ARQUIVO", "incidentes.csv")

_pacote = joblib.load("modelo_incidentes.pkl")
MODELO = _pacote["modelo"]
FEATURES = _pacote["features"]


def conectar():
    """Abre conexão com o MySQL da Azure (SSL obrigatório)."""
    return mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        ssl_disabled=False,
    )


@app.on_event("startup")
def criar_tabelas():
    """Cria as tabelas na primeira execução do container."""
    ddl_incidentes = """
        CREATE TABLE IF NOT EXISTS tb_incidentes (
            id INT AUTO_INCREMENT PRIMARY KEY,
            data_referencia DATE NOT NULL,
            qtd_incidentes INT NOT NULL,
            dt_carga TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    ddl_previsoes = """
        CREATE TABLE IF NOT EXISTS tb_previsoes (
            id INT AUTO_INCREMENT PRIMARY KEY,
            data_previsao DATE NOT NULL,
            horizonte VARCHAR(10) NOT NULL,
            qtd_prevista DECIMAL(10,2) NOT NULL,
            gerado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    ddl_log = """
        CREATE TABLE IF NOT EXISTS tb_log_execucao (
            id INT AUTO_INCREMENT PRIMARY KEY,
            evento VARCHAR(100) NOT NULL,
            detalhe VARCHAR(255),
            registrado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    try:
        con = conectar()
        cur = con.cursor()
        for ddl in (ddl_incidentes, ddl_previsoes, ddl_log):
            cur.execute(ddl)
        con.commit()
        cur.close()
        con.close()
        log.info("Tabelas verificadas/criadas com sucesso.")
    except Exception as exc:
        log.error(f"Falha ao criar tabelas: {exc}")


def registrar_log(evento: str, detalhe: str = "") -> None:
    try:
        con = conectar()
        cur = con.cursor()
        cur.execute(
            "INSERT INTO tb_log_execucao (evento, detalhe) VALUES (%s, %s)",
            (evento, detalhe[:255]),
        )
        con.commit()
        cur.close()
        con.close()
    except Exception as exc:
        log.error(f"Falha ao gravar log: {exc}")


@app.get("/")
def health():
    """Página inicial — serve de evidência de que a aplicação está no ar."""
    return {
        "aplicacao": "NexusOps - Previsao de Incidentes Locaweb",
        "status": "online",
        "modelo_carregado": True,
        "features_do_modelo": FEATURES,
    }


@app.post("/ingerir")
def ingerir():
    """PASSO 1 do fluxo: lê o CSV do Blob Storage e grava no MySQL."""
    if not STORAGE_CONN:
        raise HTTPException(500, "AZURE_STORAGE_CONNECTION_STRING nao configurada.")

    from azure.storage.blob import BlobServiceClient

    servico = BlobServiceClient.from_connection_string(STORAGE_CONN)
    blob = servico.get_blob_client(container=BLOB_CONTAINER, blob=BLOB_ARQUIVO)
    conteudo = blob.download_blob().readall().decode("utf-8")

    df = pd.read_csv(StringIO(conteudo))
    if "qtd_incidentes" not in df.columns:
        df["data"] = pd.to_datetime(df["data"])
        df = df.groupby(df["data"].dt.date).size().reset_index(name="qtd_incidentes")
        df.columns = ["data", "qtd_incidentes"]

    con = conectar()
    cur = con.cursor()
    cur.execute("TRUNCATE TABLE tb_incidentes")
    for _, linha in df.iterrows():
        cur.execute(
            "INSERT INTO tb_incidentes (data_referencia, qtd_incidentes) VALUES (%s, %s)",
            (str(linha["data"]), int(linha["qtd_incidentes"])),
        )
    con.commit()
    cur.close()
    con.close()

    registrar_log("INGESTAO", f"{len(df)} linhas carregadas do blob {BLOB_ARQUIVO}")
    log.info(f"Ingestao concluida: {len(df)} linhas.")
    return {"status": "ok", "linhas_carregadas": len(df), "origem": BLOB_ARQUIVO}


@app.post("/prever")
def prever():
    """PASSO 2 do fluxo: lê o histórico do MySQL, prevê D+1 e D+7 e grava o resultado."""
    con = conectar()
    df = pd.read_sql(
        "SELECT data_referencia, qtd_incidentes FROM tb_incidentes ORDER BY data_referencia",
        con,
    )
    con.close()

    if len(df) < 8:
        raise HTTPException(400, "Historico insuficiente. Rode /ingerir primeiro.")

    serie = df["qtd_incidentes"].tolist()
    ultima_data = pd.to_datetime(df["data_referencia"].iloc[-1]).date()

    resultados = []
    for horizonte, dias in (("D+1", 1), ("D+7", 7)):
        alvo = ultima_data + timedelta(days=dias)
        entrada = pd.DataFrame([{
            "lag_1": serie[-1],
            "lag_2": serie[-2],
            "lag_7": serie[-7],
            "media_7": sum(serie[-7:]) / 7,
            "dia_semana": alvo.weekday(),
            "fim_de_semana": int(alvo.weekday() >= 5),
        }])[FEATURES]

        previsto = float(MODELO.predict(entrada)[0])
        resultados.append({"horizonte": horizonte, "data": str(alvo), "qtd_prevista": round(previsto, 2)})

    con = conectar()
    cur = con.cursor()
    for r in resultados:
        cur.execute(
            "INSERT INTO tb_previsoes (data_previsao, horizonte, qtd_prevista) VALUES (%s, %s, %s)",
            (r["data"], r["horizonte"], r["qtd_prevista"]),
        )
    con.commit()
    cur.close()
    con.close()

    registrar_log("PREVISAO", f"{len(resultados)} previsoes geradas a partir de {ultima_data}")
    log.info(f"Previsoes geradas: {resultados}")
    return {"status": "ok", "base_ate": str(ultima_data), "previsoes": resultados}


@app.get("/previsoes")
def listar_previsoes():
    """PASSO 3 do fluxo: consulta as previsões já gravadas (evidência de leitura no banco)."""
    con = conectar()
    df = pd.read_sql(
        "SELECT data_previsao, horizonte, qtd_prevista, gerado_em "
        "FROM tb_previsoes ORDER BY gerado_em DESC LIMIT 20",
        con,
    )
    con.close()
    return {"total": len(df), "previsoes": df.astype(str).to_dict(orient="records")}


@app.get("/erro-proposital")
def erro_proposital():
    """Gera uma falha de propósito — serve para printar o erro no Application Insights."""
    raise HTTPException(500, "Erro simulado para evidenciar o monitoramento.")
