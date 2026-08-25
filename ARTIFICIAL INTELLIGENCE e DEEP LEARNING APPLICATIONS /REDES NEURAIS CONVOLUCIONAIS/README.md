# Resolução dos exercícios (CheckPoint 1) | Redes Neurais Convolucionais na prática

Projeto derivado de [`cnn/`](../cnn) para resolver **todos os exercícios de
[`cnn/exercicios.md`](../cnn/exercicios.md)**, com o desafio final pelo
**Caminho A (CIFAR-10)**. `cnn/` permanece intocado como referência.

O relatório completo (respostas do Nível 1, tabelas do Nível 2, análises do
Nível 3 e o comparativo Fashion-MNIST × CIFAR-10 do Nível 4) está publicado
como Artifact — link entregue separadamente.

## Estrutura

```
cnn_junior/
├── src/
│   ├── config.py        # Config + suporte a FashionMNIST e CIFAR-10 (RGB)
│   ├── data.py           # + flags --sem-normalizacao / --sem-augmentation
│   ├── model.py           # CNNSimples (+ usar_batchnorm) e MLPSimples (3.1)
│   ├── train.py            # CLI completa para os experimentos do Nível 2
│   ├── evaluate.py          # + acurácia top-2 (3.3), --tag por experimento
│   ├── experimentos.py       # driver: roda baseline + MLP + 12 ablações
│   ├── filtros_viz.py         # filtros e mapas de ativação (3.4)
│   ├── predict.py               # inferência em imagens novas (3.5)
│   └── export_model.py           # TorchScript (etapa 7-A)
├── deploy/                        # API FastAPI + Docker (cópia do baseline)
└── outputs/
    ├── baseline/                   # CNN completa, 12 épocas (linha de base)
    ├── mlp_baseline/                 # exercício 3.1
    ├── 2_1_sem_normalizacao/ ... 2_6_sembn_lr1e-2/   # exercício Nível 2
    ├── experimentos/resumo.json       # tabela consolidada de todos os runs
    ├── filtros/                        # exercício 3.4
    └── cifar10/                         # Nível 4 — Caminho A
```

## Reproduzir

```bash
cd cnn_junior
pip install -r requirements.txt
export KMP_DUPLICATE_LIB_OK=TRUE     # necessário neste ambiente Windows/conda

cd src
python train.py --tag baseline --epocas 12       # linha de base
python evaluate.py --tag baseline
python experimentos.py                             # todos os 14 runs do Nível 2/3.1
python filtros_viz.py --tag baseline                 # 3.4
python train.py --dataset CIFAR10 --tag cifar10_caminho_a --epocas 15   # Nível 4-A
python evaluate.py --tag cifar10_caminho_a
```

## Exercício 3.5 (fotos reais)

Não executado nesta rodada por falta de fotos reais de roupas. O comando está
pronto em `src/predict.py`:

```bash
python predict.py caminho/da/foto.jpg --tag baseline --inverter
```
