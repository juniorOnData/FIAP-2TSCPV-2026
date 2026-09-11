import json
import os
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pyflink.common import Types
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.watermark_strategy import WatermarkStrategy

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import ProcessFunction
from pyflink.datastream.output_tag import OutputTag

from pyflink.datastream.connectors.base import DeliveryGuarantee

from pyflink.datastream.connectors.kafka import (
    KafkaSource,
    KafkaSink,
    KafkaRecordSerializationSchema,
    KafkaOffsetsInitializer,
    KafkaOffsetResetStrategy,
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

KAFKA_BOOTSTRAP = "kafka:19092"

TOPIC_RAW = "ola.events.raw"
TOPIC_PREDICTIONS = "ola.predictions"
TOPIC_ALERTS = "ola.alerts"
TOPIC_DLQ = "ola.dlq"

CONSUMER_GROUP = "ola-flink-consumer"

MODEL_SERVICE_URL = os.getenv(
    "MODEL_SERVICE_URL",
    "http://model-service:8000/predict",
)

MODEL_TIMEOUT_SECONDS = int(
    os.getenv(
        "MODEL_TIMEOUT_SECONDS",
        "10",
    )
)


# ============================================================
# FEATURES DO MODELO
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

REQUIRED_FIELDS = [
    "numero",
    *MODEL_FEATURES,
]


# ============================================================
# SIDE OUTPUTS
# ============================================================

ALERT_TAG = OutputTag(
    "ola-alerts",
    Types.STRING(),
)

DLQ_TAG = OutputTag(
    "ola-dlq",
    Types.STRING(),
)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def json_dumps(data):
    return json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def validate_event(event):

    if not isinstance(event, dict):
        raise ValueError(
            "Evento deve ser um objeto JSON."
        )

    # --------------------------------------------------------
    # CAMPOS OBRIGATÓRIOS
    # --------------------------------------------------------

    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if field not in event
    ]

    if missing_fields:
        raise ValueError(
            "Campos ausentes: "
            + ", ".join(missing_fields)
        )

    # --------------------------------------------------------
    # NUMERO
    # --------------------------------------------------------

    numero = event.get("numero")

    if (
        not isinstance(numero, str)
        or not numero.strip()
    ):
        raise ValueError(
            "Campo 'numero' deve ser uma string não vazia."
        )

    # --------------------------------------------------------
    # CATEGÓRICAS
    # --------------------------------------------------------

    categorical_fields = [
        "tipo_alerta",
        "origem_abertura",
        "produto",
        "item_configuracao",
        "prioridade_origem",
        "dia_semana_abertura",
    ]

    for field in categorical_fields:

        value = event.get(field)

        if (
            value is not None
            and not isinstance(value, str)
        ):
            raise ValueError(
                f"Campo '{field}' deve ser string ou null."
            )

    # --------------------------------------------------------
    # NUMÉRICAS
    # --------------------------------------------------------

    numeric_fields = [
        "prioridade_cod",
        "hora_abertura",
    ]

    for field in numeric_fields:

        value = event.get(field)

        if value is not None:

            if (
                isinstance(value, bool)
                or not isinstance(
                    value,
                    (int, float),
                )
            ):
                raise ValueError(
                    f"Campo '{field}' deve ser numérico ou null."
                )

    # --------------------------------------------------------
    # BOOLEANAS
    # --------------------------------------------------------

    boolean_fields = [
        "fl_tem_incidente_pai",
        "fl_automatico",
    ]

    for field in boolean_fields:

        value = event.get(field)

        if (
            value is not None
            and not isinstance(value, bool)
        ):
            raise ValueError(
                f"Campo '{field}' deve ser boolean ou null."
            )


def build_dlq_event(
    raw_event,
    stage,
    error_type,
    error_message,
):

    return {
        "dlq_timestamp": utc_now(),
        "stage": stage,
        "error_type": error_type,
        "error_message": error_message,
        "source_topic": TOPIC_RAW,
        "original_event": raw_event,
    }


# ============================================================
# PROCESS FUNCTION
# ============================================================

class InferenceProcessFunction(
    ProcessFunction
):

    def process_element(
        self,
        value,
        ctx,
    ):

        # ====================================================
        # 1. PARSE JSON
        # ====================================================

        try:

            event = json.loads(value)

        except Exception as exc:

            dlq = build_dlq_event(
                raw_event=value,
                stage="json_parse",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

            yield (
                DLQ_TAG,
                json_dumps(dlq),
            )

            return

        # ====================================================
        # 2. VALIDAÇÃO DO SCHEMA
        # ====================================================

        try:

            validate_event(event)

        except Exception as exc:

            dlq = build_dlq_event(
                raw_event=event,
                stage="schema_validation",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

            yield (
                DLQ_TAG,
                json_dumps(dlq),
            )

            return

        # ====================================================
        # 3. PAYLOAD PARA MODEL-SERVICE
        # ====================================================

        request_payload = {
            "numero": event["numero"],
        }

        for feature in MODEL_FEATURES:

            request_payload[feature] = (
                event.get(feature)
            )

        # ====================================================
        # 4. CHAMADA AO MODELO
        # ====================================================

        try:

            request = Request(
                MODEL_SERVICE_URL,
                data=json_dumps(
                    request_payload
                ).encode("utf-8"),
                headers={
                    "Content-Type":
                        "application/json"
                },
                method="POST",
            )

            with urlopen(
                request,
                timeout=MODEL_TIMEOUT_SECONDS,
            ) as response:

                response_body = (
                    response
                    .read()
                    .decode("utf-8")
                )

            model_result = json.loads(
                response_body
            )

        except HTTPError as exc:

            try:
                error_body = (
                    exc.read()
                    .decode(
                        "utf-8",
                        errors="replace",
                    )
                )

            except Exception:
                error_body = ""

            dlq = build_dlq_event(
                raw_event=event,
                stage="model_inference",
                error_type="HTTPError",
                error_message=(
                    f"HTTP {exc.code}: "
                    f"{error_body}"
                ),
            )

            yield (
                DLQ_TAG,
                json_dumps(dlq),
            )

            return

        except (
            URLError,
            TimeoutError,
            Exception,
        ) as exc:

            dlq = build_dlq_event(
                raw_event=event,
                stage="model_inference",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

            yield (
                DLQ_TAG,
                json_dumps(dlq),
            )

            return

        # ====================================================
        # 5. VALIDA RESPOSTA DO MODELO
        # ====================================================

        try:

            required_response_fields = [
                "numero",
                "probabilidade_nao_violacao",
                "probabilidade_violacao",
                "threshold_alerta",
                "alerta",
                "model_uri",
                "prediction_timestamp",
            ]

            missing_response = [
                field
                for field
                in required_response_fields
                if field not in model_result
            ]

            if missing_response:

                raise ValueError(
                    "Resposta do modelo sem campos: "
                    + ", ".join(
                        missing_response
                    )
                )

        except Exception as exc:

            dlq = build_dlq_event(
                raw_event=event,
                stage="model_response_validation",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

            yield (
                DLQ_TAG,
                json_dumps(dlq),
            )

            return

        # ====================================================
        # 6. MONTA EVENTO DE PREDIÇÃO
        # ====================================================

        prediction_event = {

            # ----------------------------------------------
            # Identificação
            # ----------------------------------------------

            "numero":
                model_result["numero"],

            # ----------------------------------------------
            # Metadados originais opcionais
            # ----------------------------------------------

            "event_id":
                event.get("event_id"),

            "event_type":
                event.get("event_type"),

            "event_timestamp":
                event.get("event_timestamp"),

            # ----------------------------------------------
            # Resultado ML
            # ----------------------------------------------

            "probabilidade_nao_violacao":
                model_result[
                    "probabilidade_nao_violacao"
                ],

            "probabilidade_violacao":
                model_result[
                    "probabilidade_violacao"
                ],

            "threshold_alerta":
                model_result[
                    "threshold_alerta"
                ],

            "alerta":
                model_result["alerta"],

            # ----------------------------------------------
            # Modelo
            # ----------------------------------------------

            "model_uri":
                model_result["model_uri"],

            "prediction_timestamp":
                model_result[
                    "prediction_timestamp"
                ],

            # ----------------------------------------------
            # Pipeline
            # ----------------------------------------------

            "stream_processing_timestamp":
                utc_now(),

            "source_topic":
                TOPIC_RAW,
        }

        prediction_json = json_dumps(
            prediction_event
        )

        # ====================================================
        # 7. TODAS AS PREVISÕES
        # ====================================================

        yield prediction_json

        # ====================================================
        # 8. SOMENTE ALERTAS
        # ====================================================

        if bool(
            model_result["alerta"]
        ):

            yield (
                ALERT_TAG,
                prediction_json,
            )


# ============================================================
# KAFKA SINK
# ============================================================

def create_kafka_sink(topic):

    serializer = (
        KafkaRecordSerializationSchema
        .builder()
        .set_topic(topic)
        .set_value_serialization_schema(
            SimpleStringSchema()
        )
        .build()
    )

    return (
        KafkaSink
        .builder()
        .set_bootstrap_servers(
            KAFKA_BOOTSTRAP
        )
        .set_record_serializer(
            serializer
        )
        .set_delivery_guarantee(
            DeliveryGuarantee.AT_LEAST_ONCE
        )
        .build()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # ENVIRONMENT
    # ========================================================

    env = (
        StreamExecutionEnvironment
        .get_execution_environment()
    )

    env.set_parallelism(2)

    # Checkpoint a cada 10 segundos
    env.enable_checkpointing(
        10_000
    )

    # ========================================================
    # KAFKA SOURCE
    # ========================================================

    kafka_source = (
        KafkaSource
        .builder()

        .set_bootstrap_servers(
            KAFKA_BOOTSTRAP
        )

        .set_topics(
            TOPIC_RAW
        )

        .set_group_id(
            CONSUMER_GROUP
        )

        # Usa o último offset salvo pelo grupo.
        # Se não existir, começa do início.
        .set_starting_offsets(
            KafkaOffsetsInitializer
            .committed_offsets(
                KafkaOffsetResetStrategy
                .EARLIEST
            )
        )

        .set_value_only_deserializer(
            SimpleStringSchema()
        )

        .build()
    )

    raw_stream = env.from_source(
        kafka_source,
        WatermarkStrategy.no_watermarks(),
        "Kafka Source - ola.events.raw",
    )

    # ========================================================
    # INFERÊNCIA
    # ========================================================

    predictions_stream = (
        raw_stream
        .process(
            InferenceProcessFunction(),
            output_type=Types.STRING(),
        )
        .name(
            "KPI Violation Inference"
        )
    )

    # ========================================================
    # SIDE OUTPUTS
    # ========================================================

    alerts_stream = (
        predictions_stream
        .get_side_output(
            ALERT_TAG
        )
    )

    dlq_stream = (
        predictions_stream
        .get_side_output(
            DLQ_TAG
        )
    )

    # ========================================================
    # SINK - PREDICTIONS
    # ========================================================

    predictions_stream.sink_to(
        create_kafka_sink(
            TOPIC_PREDICTIONS
        )
    ).name(
        "Kafka Sink - ola.predictions"
    )

    # ========================================================
    # SINK - ALERTS
    # ========================================================

    alerts_stream.sink_to(
        create_kafka_sink(
            TOPIC_ALERTS
        )
    ).name(
        "Kafka Sink - ola.alerts"
    )

    # ========================================================
    # SINK - DLQ
    # ========================================================

    dlq_stream.sink_to(
        create_kafka_sink(
            TOPIC_DLQ
        )
    ).name(
        "Kafka Sink - ola.dlq"
    )

    # ========================================================
    # EXECUÇÃO
    # ========================================================

    env.execute(
        "OLA Streaming - KPI Violation Prediction"
    )


if __name__ == "__main__":
    main()
