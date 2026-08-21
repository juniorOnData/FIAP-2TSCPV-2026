# Comece Aqui

Este repositório reúne o código, a infraestrutura e a documentação técnica desenvolvidos para a disciplina de **Cloud Solutions & Scalable Infrastructure**, durante a Sprint 3 do FIAP Challenge 2026.

A documentação está organizada para apresentar primeiro a arquitetura da solução, depois os recursos utilizados, a execução realizada, o monitoramento, os problemas encontrados e, por fim, os resultados obtidos.

## Navegação

A leitura recomendada segue a ordem abaixo.

### 1. Visão geral

O `README.md` apresenta o projeto e seus principais componentes.

### 2. Arquitetura

Apresenta a arquitetura da solução, os componentes Azure utilizados e o fluxo de dados entre aplicação, banco, armazenamento e monitoramento.

### 3. Recursos Azure

Registra os recursos provisionados, suas configurações e a organização da infraestrutura.

### 4. Execução

Documenta a execução realizada durante o provisionamento e publicação da aplicação, incluindo os principais comandos utilizados e as validações executadas.

### 5. Monitoramento

Apresenta a configuração de logs, métricas, alertas e Application Insights utilizada no ambiente.

### 6. Troubleshooting

Registra os principais problemas encontrados durante a implementação e as soluções utilizadas.

### 7. Relatório final

Consolida os resultados da implementação, os testes realizados, os desafios encontrados e as limitações conhecidas da solução.

## Estrutura do repositório

```text
ENTERPRISE CHALLENGE SPRINT 3/
│
├── README.md
├── COMECE_AQUI.md
│
├── docs/
│   ├── 1. Arquitetura.md
│   ├── 2. Recursos Azure.md
│   ├── 3. Execução.md
│   ├── 4. Monitoramento.md
│   ├── 5. Troubleshooting.md
│   └── 6. Relatório final.md
│
├── src/
│   ├── app.py
│   └── requirements.txt
│
├── docker/
│   └── Dockerfile
│
├── scripts/
│   ├── deploy_azure.sh
│   └── treinar_modelo.py
│
├── data/
│   └── sample_incidentes.csv
│
├── evidencias/
│   └── screenshots da execução
│
└── .gitignore
