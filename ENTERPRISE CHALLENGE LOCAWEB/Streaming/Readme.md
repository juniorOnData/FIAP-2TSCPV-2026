# Streaming Data Architecture for KPI Violation Prediction

Arquitetura de dados em streaming desenvolvida com **Apache Kafka**, **Apache Flink**, **Docker**, **FastAPI**, **MLflow** e **Databricks** para realizar inferência em tempo real sobre o risco de violação de KPI em incidentes de TI.

O projeto integra processamento de eventos em streaming com um modelo de Machine Learning previamente treinado no Databricks e registrado no **MLflow Model Registry / Unity Catalog**.

---

## Objetivo

Construir uma arquitetura capaz de:

- Receber eventos de incidentes em tempo real;
- Utilizar Apache Kafka como barramento de eventos;
- Processar os eventos utilizando Apache Flink;
- Validar os dados recebidos;
- Realizar inferência utilizando um modelo de Machine Learning;
- Recuperar automaticamente o modelo `champion` registrado no MLflow;
- Publicar todas as previsões em um tópico Kafka;
- Gerar alertas para incidentes com alta probabilidade de violação;
- Encaminhar eventos inválidos para uma Dead Letter Queue.

---

## Arquitetura

```mermaid
flowchart LR

    A[Evento de Incidente] --> B[Apache Kafka<br/>ola.events.raw]

    B --> C[Apache Flink]

    C --> D[Validação JSON / Schema]

    D --> E[Model Service<br/>FastAPI]

    E --> F[MLflow / Unity Catalog<br/>Databricks]

    F --> G[previsao_kpi_gbm<br/>@champion]

    G --> E

    E --> C

    C --> H[ola.predictions]

    C --> I[ola.alerts]

    D --> J[ola.dlq]
```

### Fluxo simplificado

```text
Evento
   ↓
Apache Kafka
   ↓
ola.events.raw
   ↓
Apache Flink
   ↓
Validação
   ↓
Model Service
   ↓
MLflow / Databricks
   ↓
Modelo Champion
   ↓
Probabilidade de Violação
   ↓
Apache Flink
   │
   ├── ola.predictions
   ├── ola.alerts
   └── ola.dlq
```

---

## Tecnologias utilizadas

- Apache Kafka 4.3.1
- Apache Flink 2.1.3
- PyFlink
- Docker
- Docker Compose
- Python
- FastAPI
- MLflow
- Databricks
- Unity Catalog
- Scikit-learn
- Pandas
- Kafka UI

---

## Apache Kafka

O Apache Kafka funciona como barramento de eventos da solução.

Foram criados quatro tópicos principais:

| Tópico | Finalidade |
|---|---|
| `ola.events.raw` | Recebe os eventos de incidentes |
| `ola.predictions` | Armazena todas as previsões |
| `ola.alerts` | Recebe previsões que ultrapassaram o threshold de alerta |
| `ola.dlq` | Recebe eventos que apresentaram erro |

### Consumer Group

O Apache Flink utiliza o consumer group:

```text
ola-flink-consumer
```

---

## Apache Flink

O Apache Flink é responsável pelo processamento dos eventos em streaming.

O job principal da aplicação é:

```text
OLA Streaming - KPI Violation Prediction
```

O processamento executa as seguintes etapas:

```text
Kafka
  ↓
Leitura do evento
  ↓
Conversão do JSON
  ↓
Validação do schema
  ↓
Separação das features
  ↓
Requisição ao Model Service
  ↓
Recebimento da previsão
  ↓
Publicação no Kafka
```

O Flink também utiliza **checkpointing**, permitindo recuperação do processamento em caso de falhas.

```text
Checkpoint Interval: 10 segundos
```

---

## Machine Learning

O modelo de Machine Learning foi previamente treinado no Databricks.

O algoritmo utilizado é baseado em:

```text
HistGradientBoostingClassifier
```

O pipeline também contém o pré-processamento necessário para as variáveis utilizadas na inferência.

O modelo está registrado no Unity Catalog com o nome:

```text
fiap_analytics.ml.previsao_kpi_gbm
```

A aplicação utiliza o alias:

```text
@champion
```

Portanto, o modelo é carregado utilizando:

```text
models:/fiap_analytics.ml.previsao_kpi_gbm@champion
```

Essa estratégia permite alterar qual versão é considerada `champion` no MLflow sem precisar alterar o código da aplicação.

---

## Features utilizadas pelo modelo

O modelo recebe 10 features:

### Variáveis categóricas

```text
tipo_alerta
origem_abertura
produto
item_configuracao
prioridade_origem
dia_semana_abertura
```

### Variáveis numéricas

```text
prioridade_cod
hora_abertura
```

### Variáveis booleanas

```text
fl_tem_incidente_pai
fl_automatico
```

O pré-processamento dessas variáveis está encapsulado no próprio pipeline registrado no MLflow.

---

## Model Service

O modelo é executado em um serviço separado utilizando **FastAPI**.

Essa separação permite manter o ambiente necessário para Machine Learning independente do ambiente utilizado pelo PyFlink.

```text
Apache Flink
      │
      │ HTTP
      ▼
Model Service
      │
      ▼
MLflow
      │
      ▼
Modelo Champion
      │
      ▼
predict_proba()
```

O Model Service utiliza um ambiente compatível com o modelo treinado:

```text
Python 3.12
MLflow 3.8.1
Scikit-learn 1.6.1
Cloudpickle 3.0.0
```

---

## Endpoints

### Health Check

```http
GET /health
```

Exemplo de resposta:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_uri": "models:/fiap_analytics.ml.previsao_kpi_gbm@champion",
  "threshold": 0.8958
}
```

---

### Inferência

```http
POST /predict
```

Exemplo de entrada:

```json
{
  "numero": "INC_STREAM_001",
  "tipo_alerta": "Monitoramento",
  "origem_abertura": "Automatica",
  "produto": "Hospedagem",
  "item_configuracao": "Servidor Web",
  "prioridade_origem": "P2",
  "dia_semana_abertura": "Segunda-feira",
  "prioridade_cod": 2,
  "hora_abertura": 14,
  "fl_tem_incidente_pai": false,
  "fl_automatico": true
}
```

Exemplo de saída:

```json
{
  "numero": "INC_STREAM_001",
  "probabilidade_nao_violacao": 0.684301,
  "probabilidade_violacao": 0.315699,
  "threshold_alerta": 0.8958,
  "alerta": false,
  "model_uri": "models:/fiap_analytics.ml.previsao_kpi_gbm@champion"
}
```

---

## Threshold de alerta

O modelo retorna a probabilidade de violação.

A decisão de gerar um alerta utiliza o threshold:

```text
0.8958
```

A regra aplicada é:

```text
probabilidade_violacao >= 0.8958
                ↓
          alerta = true
```

Caso contrário:

```text
alerta = false
```

O threshold é configurado através de uma variável de ambiente:

```env
ALERT_THRESHOLD=0.8958
```

---

## Resultado do teste ponta a ponta

A arquitetura foi validada através de um evento publicado no tópico:

```text
ola.events.raw
```

Evento:

```text
INC_STREAM_001
```

O evento percorreu todo o pipeline:

```text
Kafka
  ↓
Flink
  ↓
Model Service
  ↓
MLflow / Databricks
  ↓
Modelo Champion
  ↓
Flink
  ↓
Kafka
```

Resultado da inferência:

```text
Probabilidade de não violação: 68,43%
Probabilidade de violação:     31,57%

Threshold de alerta:           89,58%

Alerta:                        False
```

O resultado foi publicado com sucesso no tópico:

```text
ola.predictions
```

Mensagem publicada:

```json
{
  "numero": "INC_STREAM_001",
  "event_id": "EVENT_001",
  "event_type": "INCIDENT_OPENED",
  "event_timestamp": "2026-09-09T01:35:00+00:00",
  "probabilidade_nao_violacao": 0.684301,
  "probabilidade_violacao": 0.315699,
  "threshold_alerta": 0.8958,
  "alerta": false,
  "model_uri": "models:/fiap_analytics.ml.previsao_kpi_gbm@champion",
  "source_topic": "ola.events.raw"
}
```

---

## Dead Letter Queue

Eventos que apresentam problema durante o processamento são direcionados para:

```text
ola.dlq
```

Entre os erros tratados estão:

- JSON inválido;
- Campos obrigatórios ausentes;
- Tipos de dados inválidos;
- Erro na comunicação com o Model Service;
- Erro durante a inferência;
- Resposta inválida do serviço de Machine Learning.

Dessa forma, um evento problemático não interrompe o processamento do restante do stream.

---

## Estrutura do projeto

```text
streaming-ola/
│
├── docker-compose.yml
├── .env.example
├── .gitignore
├── README.md
│
├── flink/
│   ├── Dockerfile
│   ├── requirements.txt
│   │
│   └── jobs/
│       └── ola_streaming.py
│
└── model-service/
    ├── Dockerfile
    ├── requirements.txt
    └── app.py
```

---

## Containers

Toda a arquitetura é executada através do Docker Compose.

Os serviços utilizados são:

```text
kafka
kafka-init
kafka-ui

flink-init
flink-jobmanager
flink-taskmanager

model-service
```

---

## Interfaces

### Kafka UI

```text
http://localhost:8080
```

Permite acompanhar:

- Tópicos;
- Mensagens;
- Partições;
- Consumer Groups;
- Offsets;
- Lag.

---

### Apache Flink

```text
http://localhost:8081
```

Permite acompanhar:

- Jobs em execução;
- TaskManagers;
- Task Slots;
- Checkpoints;
- Métricas;
- Falhas do pipeline.

---

### FastAPI / Model Service

Health Check:

```text
http://localhost:8000/health
```

Swagger:

```text
http://localhost:8000/docs
```

---

## Docker

Toda a solução foi containerizada.

```text
Docker
│
├── Apache Kafka
│
├── Kafka UI
│
├── Flink JobManager
│
├── Flink TaskManager
│
└── Model Service
```

Os containers compartilham uma rede Docker:

```text
ola-network
```

Isso permite a comunicação interna utilizando nomes de serviço, por exemplo:

```text
kafka:19092
```

e:

```text
http://model-service:8000
```

---

## Segurança

As credenciais do Databricks não são armazenadas no código.

São fornecidas através de variáveis de ambiente.

Exemplo:

```env
DATABRICKS_HOST=https://seu-workspace.cloud.databricks.com
DATABRICKS_TOKEN=SEU_TOKEN
ALERT_THRESHOLD=0.8958
```

O arquivo real:

```text
.env
```

não deve ser versionado no GitHub.

O repositório disponibiliza apenas:

```text
.env.example
```

---

## Execução

Para subir toda a arquitetura:

```bash
docker compose up -d --build
```

Verificar os containers:

```bash
docker compose ps -a
```

Verificar os tópicos Kafka:

```bash
docker exec kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server kafka:19092 \
  --list
```

Submeter o job do Flink:

```bash
docker exec flink-jobmanager \
  flink run -d \
  -py /opt/flink/jobs/ola_streaming.py
```

---

## Status

| Componente | Status |
|---|---|
| Apache Kafka | ✅ |
| Kafka UI | ✅ |
| Tópicos Kafka | ✅ |
| Apache Flink | ✅ |
| PyFlink | ✅ |
| JobManager | ✅ |
| TaskManager | ✅ |
| Checkpointing | ✅ |
| Kafka → Flink | ✅ |
| MLflow | ✅ |
| Unity Catalog | ✅ |
| Modelo `@champion` | ✅ |
| Model Service | ✅ |
| FastAPI | ✅ |
| Endpoint `/health` | ✅ |
| Endpoint `/predict` | ✅ |
| Flink → Model Service | ✅ |
| Model Service → MLflow | ✅ |
| `ola.predictions` | ✅ |
| Pipeline ponta a ponta | ✅ |

---

## Possíveis evoluções

A arquitetura pode ser evoluída com:

- Schema Registry;
- Avro ou Protobuf;
- Inferência assíncrona no Apache Flink;
- Controle de idempotência através de `event_id`;
- Observabilidade com Prometheus e Grafana;
- Persistência das previsões em Lakehouse;
- Monitoramento de Data Drift;
- Monitoramento de Model Drift;
- Feature Store;
- Retreinamento automatizado;
- Atualização automática do modelo `champion`;
- Orquestração de pipelines;
- Integração com sistemas de notificação para alertas críticos.

---

## Conclusão

A POC demonstrou a integração entre processamento de eventos em streaming e Machine Learning.

A arquitetura conseguiu executar com sucesso o fluxo completo:

```text
Evento
   ↓
Apache Kafka
   ↓
Apache Flink
   ↓
FastAPI
   ↓
MLflow / Databricks
   ↓
Modelo de Machine Learning
   ↓
Apache Flink
   ↓
Kafka
```

A solução permite que novos eventos sejam processados em tempo real e classificados de acordo com sua probabilidade de violação, criando uma base para sistemas de monitoramento e alertas proativos.
