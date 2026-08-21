"""
data.py — ETAPAS 1 e 2 do pipeline: aquisição e pré-processamento
------------------------------------------------------------------
Extensão de cnn/src/data.py para suportar os experimentos dos exercícios:

  * sem_normalizacao  (2.1) -> remove transforms.Normalize
  * sem_augmentation  (2.2) -> transform de treino = transform de avaliação
  * dataset="CIFAR10" (4-A) -> troca Fashion-MNIST por CIFAR-10 (colorido, 32x32)

Regra de ouro mantida: augmentation só no treino; validação e teste sempre
determinísticos; treino/validação carregados como cópias com Subset para não
compartilhar transform (ver comentário original abaixo).
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import CFG, DATA_DIR, config_para_dataset, classes_do_dataset


def construir_transformacoes(cfg, sem_normalizacao: bool = False,
                             sem_augmentation: bool = False):
    """Devolve (transform_treino, transform_avaliacao)."""
    passos_norm = [] if sem_normalizacao else [transforms.Normalize(cfg.media, cfg.desvio)]

    if sem_augmentation:
        passos_treino = [transforms.ToTensor(), *passos_norm]
    else:
        passos_treino = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(cfg.tamanho_imagem, padding=2),
            transforms.RandomRotation(degrees=8),
            transforms.ToTensor(),
            *passos_norm,
        ]

    transform_treino = transforms.Compose(passos_treino)
    transform_avaliacao = transforms.Compose([transforms.ToTensor(), *passos_norm])
    return transform_treino, transform_avaliacao


def _classe_dataset(nome: str):
    return {"FashionMNIST": datasets.FashionMNIST, "CIFAR10": datasets.CIFAR10}[nome]


def obter_dataloaders(cfg=CFG, rapido: bool = False, sem_normalizacao: bool = False,
                      sem_augmentation: bool = False):
    """Baixa o dataset (cfg.dataset) e devolve os três DataLoaders + classes."""
    t_treino, t_aval = construir_transformacoes(cfg, sem_normalizacao, sem_augmentation)
    Dataset = _classe_dataset(cfg.dataset)
    classes = classes_do_dataset(cfg.dataset)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    base_treino_aug = Dataset(root=DATA_DIR, train=True, download=True, transform=t_treino)
    base_treino_limpo = Dataset(root=DATA_DIR, train=True, download=True, transform=t_aval)
    conjunto_teste = Dataset(root=DATA_DIR, train=False, download=True, transform=t_aval)

    n_total = len(base_treino_aug)
    gerador = torch.Generator().manual_seed(cfg.semente)
    indices = torch.randperm(n_total, generator=gerador).tolist()

    n_val = int(n_total * cfg.frac_validacao)
    idx_val, idx_treino = indices[:n_val], indices[n_val:]

    if rapido:
        idx_treino, idx_val = idx_treino[:6000], idx_val[:1000]
        conjunto_teste = Subset(conjunto_teste, list(range(2000)))

    conjunto_treino = Subset(base_treino_aug, idx_treino)
    conjunto_val = Subset(base_treino_limpo, idx_val)

    args_comuns = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                       pin_memory=torch.cuda.is_available())

    carregador_treino = DataLoader(conjunto_treino, shuffle=True, drop_last=False, **args_comuns)
    carregador_val = DataLoader(conjunto_val, shuffle=False, **args_comuns)
    carregador_teste = DataLoader(conjunto_teste, shuffle=False, **args_comuns)

    return carregador_treino, carregador_val, carregador_teste, classes


if __name__ == "__main__":
    from config import OUT_DIR
    from utils import salvar_amostra

    p = argparse.ArgumentParser(description="Inspeção do dataset")
    p.add_argument("--rapido", action="store_true")
    p.add_argument("--dataset", default="FashionMNIST", choices=["FashionMNIST", "CIFAR10"])
    args = p.parse_args()

    cfg = config_para_dataset(args.dataset)
    tr, va, te, classes = obter_dataloaders(cfg, rapido=args.rapido)

    print(f"Dataset: {cfg.dataset}")
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Treino    : {len(tr.dataset):>6} imagens | {len(tr)} lotes")
    print(f"Validação : {len(va.dataset):>6} imagens | {len(va)} lotes")
    print(f"Teste     : {len(te.dataset):>6} imagens | {len(te)} lotes")

    x, y = next(iter(tr))
    print(f"\nFormato do lote x: {tuple(x.shape)}  (N, C, H, W)")
    print(f"Formato do lote y: {tuple(y.shape)}  dtype={y.dtype}")
    print(f"Faixa de valores após normalizar: [{x.min():.2f}, {x.max():.2f}]")

    salvar_amostra(tr, classes, OUT_DIR / f"amostra_treino_{cfg.dataset.lower()}.png",
                   cfg.media, cfg.desvio)
