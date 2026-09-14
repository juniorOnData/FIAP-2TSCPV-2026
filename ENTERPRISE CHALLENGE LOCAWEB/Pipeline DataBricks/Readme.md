# 🌌 DataGalaxy

### AIOps para Previsão de Incidentes e Riscos Operacionais

Projeto desenvolvido pela equipe **Nexus Ops** para o **Challenge FIAP + Locaweb 2026**.

O **DataGalaxy** é uma solução de AIOps voltada à análise de incidentes de TI, utilizando técnicas de Ciência de Dados e Machine Learning para identificar padrões, antecipar comportamentos operacionais e apoiar a tomada de decisão.

---

## 🎯 Objetivo

O projeto busca transformar dados históricos de incidentes em informações preditivas capazes de apoiar operações de TI de forma preventiva.

Entre os principais objetivos estão:

- prever o volume futuro de incidentes;
- identificar risco de violação de indicadores operacionais;
- analisar o risco de descumprimento de SLA/OLA;
- identificar padrões e agrupamentos nos incidentes;
- gerar informações para apoiar ações preventivas.

---

## 📂 Notebooks

| Notebook | Descrição |
|---|---|
| `K-Means.ipynb` | Aplicação de técnicas de clusterização para identificação de grupos e padrões nos incidentes. |
| `PREVISAO_VIOLACAO_KPI.ipynb` | Desenvolvimento da abordagem preditiva para estimativa do risco de violação do KPI operacional. |
| `Previsão de Volume.ipynb` | Análise e modelagem voltadas à previsão do volume futuro de incidentes. |
| `Risco de Violação de SLA.ipynb` | Análise do risco de descumprimento dos níveis de serviço associados aos incidentes. |

---

## 🧠 Abordagem Analítica

O desenvolvimento do DataGalaxy contempla diferentes frentes de Machine Learning:

### Previsão de Volume

Utilização dos dados históricos para identificar comportamento temporal e estimar a quantidade futura de incidentes, auxiliando no planejamento da operação.

### Previsão de Violação

Construção de modelos capazes de estimar a probabilidade de um incidente apresentar risco de violação de indicadores operacionais.

### Análise de SLA/OLA

Avaliação dos incidentes com foco no cumprimento dos tempos estabelecidos para atendimento e resolução.

### Clusterização

Aplicação do algoritmo **K-Means** para identificar agrupamentos de incidentes com características semelhantes e apoiar a descoberta de padrões operacionais.

---

## 🔄 Visão Geral

```text
Dados Históricos de Incidentes
            │
            ▼
   Preparação dos Dados
            │
            ▼
   Engenharia de Features
            │
      ┌─────┴─────┐
      │           │
      ▼           ▼
 Machine       K-Means
 Learning    Clusterização
      │
      ▼
 ┌───────────────────────────┐
 │ Previsão de Volume        │
 │ Risco de Violação de KPI  │
 │ Risco de SLA / OLA        │
 └───────────────────────────┘
      │
      ▼
 Apoio à Tomada de Decisão
