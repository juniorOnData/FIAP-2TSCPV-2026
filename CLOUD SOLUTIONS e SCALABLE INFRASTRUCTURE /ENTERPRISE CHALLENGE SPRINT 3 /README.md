# Cloud Solutions & Scalable Infrastructure - FIAP Challenge 2026

## Visão geral

Este projeto apresenta a implementação da infraestrutura Cloud do Data Galaxy, com foco na execução do pipeline de processamento de incidentes em um ambiente Microsoft Azure.

A solução utiliza uma aplicação FastAPI containerizada, banco de dados gerenciado, armazenamento de objetos e serviços de monitoramento. A infraestrutura foi provisionada e validada durante a execução do projeto, com os principais fluxos da aplicação funcionando no ambiente Azure.

A arquitetura e os detalhes da implementação estão documentados separadamente em `docs/`.

## Arquitetura

A solução utiliza os seguintes componentes:

```text
                         Internet
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Azure Container      │
                 │ Instance             │
                 │                     │
                 │ FastAPI :8000       │
                 └──────────┬──────────┘
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
          MySQL          Blob Storage    Monitoring
       db_nexusops       raw/processed   App Insights
                         logs/            Log Analytics
