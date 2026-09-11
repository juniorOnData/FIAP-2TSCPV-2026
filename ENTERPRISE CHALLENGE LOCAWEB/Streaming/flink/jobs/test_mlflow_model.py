import os
import mlflow


MODEL_URI = (
    "models:/fiap_analytics.ml.previsao_kpi_gbm@champion"
)


def main():

    print("=" * 60)
    print("TESTE DE CONEXAO COM DATABRICKS / MLFLOW")
    print("=" * 60)

    # --------------------------------------------------
    # Verifica credenciais
    # --------------------------------------------------

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

    print("DATABRICKS_HOST encontrado.")
    print("DATABRICKS_TOKEN encontrado.")

    # NUNCA imprimir o token

    # --------------------------------------------------
    # Configuração MLflow
    # --------------------------------------------------

    mlflow.set_tracking_uri("databricks")

    mlflow.set_registry_uri("databricks-uc")

    print()
    print(f"Carregando modelo:")
    print(MODEL_URI)
    print()

    # --------------------------------------------------
    # Carrega modelo
    # --------------------------------------------------

    model = mlflow.pyfunc.load_model(
        MODEL_URI
    )

    print("=" * 60)
    print("MODELO CARREGADO COM SUCESSO")
    print("=" * 60)

    # --------------------------------------------------
    # Signature
    # --------------------------------------------------

    signature = model.metadata.signature

    print()
    print("Signature do modelo:")
    print(signature)

    print()
    print("Schema de entrada:")

    if signature is not None:
        print(signature.inputs)

    print()
    print("Schema de saida:")

    if signature is not None:
        print(signature.outputs)


if __name__ == "__main__":
    main()