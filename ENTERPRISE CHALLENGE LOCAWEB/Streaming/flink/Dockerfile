FROM flink:2.1.3-java17

USER root

# =========================================================
# PYTHON
# =========================================================

RUN apt-get update \
    && apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        curl \
    && rm -rf /var/lib/apt/lists/*


# =========================================================
# VIRTUAL ENVIRONMENT PYFLINK
# =========================================================

RUN python3 -m venv /opt/pyflink

ENV PATH="/opt/pyflink/bin:${PATH}"

ENV PYFLINK_PYTHON="/opt/pyflink/bin/python"
ENV PYFLINK_CLIENT_EXECUTABLE="/opt/pyflink/bin/python"


# =========================================================
# DEPENDÊNCIAS PYTHON
# =========================================================

COPY requirements.txt /tmp/requirements.txt

RUN pip install \
    --no-cache-dir \
    --upgrade pip \
    && pip install \
    --no-cache-dir \
    -r /tmp/requirements.txt


# =========================================================
# FLINK KAFKA CONNECTOR
# =========================================================

RUN curl -fSL \
    https://repo.maven.apache.org/maven2/org/apache/flink/flink-sql-connector-kafka/5.0.0-2.1/flink-sql-connector-kafka-5.0.0-2.1.jar \
    -o /opt/flink/lib/flink-sql-connector-kafka-5.0.0-2.1.jar


# =========================================================
# DIRETÓRIOS DO PROJETO
# =========================================================

RUN mkdir -p \
    /opt/flink/jobs \
    /opt/flink/checkpoints \
    /opt/flink/savepoints


WORKDIR /opt/flink