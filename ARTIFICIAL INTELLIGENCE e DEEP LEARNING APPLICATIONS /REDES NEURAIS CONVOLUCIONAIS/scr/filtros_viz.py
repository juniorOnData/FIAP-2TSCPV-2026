"""
filtros_viz.py — exercício 3.4: visualização de filtros e mapas de ativação
-----------------------------------------------------------------------------
1. Salva os filtros 3x3 da PRIMEIRA convolução (modelo.extrator[0][0].weight)
   como um mosaico de imagens — o que a rede "procura" na entrada bruta.
2. Salva os mapas de ativação do primeiro bloco convolucional para uma
   imagem de teste — onde cada filtro respondeu forte.

Uso:
    python filtros_viz.py --tag ""              # baseline (outputs/)
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from config import OUT_DIR, config_para_dataset
from data import obter_dataloaders
from evaluate import carregar_checkpoint
from utils import obter_dispositivo, _desnormalizar


def salvar_filtros(modelo, caminho):
    """Filtros da primeira Conv2d: shape (out_channels, in_channels, 3, 3)."""
    pesos = modelo.extrator[0][0].weight.detach().cpu()
    n = pesos.shape[0]
    canais_entrada = pesos.shape[1]

    colunas = 8
    linhas = (n + colunas - 1) // colunas
    fig, eixos = plt.subplots(linhas, colunas, figsize=(colunas * 1.3, linhas * 1.3))
    for i, ax in enumerate(eixos.ravel()):
        if i < n:
            f = pesos[i]
            if canais_entrada == 1:
                img = f.squeeze(0).numpy()
                ax.imshow(img, cmap="gray")
            else:
                img = f.permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                ax.imshow(img)
            ax.set_title(f"f{i}", fontsize=7)
        ax.axis("off")
    fig.suptitle(f"Filtros 3x3 da 1ª convolução ({n} filtros)")
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)
    print(f"[ok] Filtros salvos em {caminho}")


def salvar_ativacoes(modelo, imagem, rotulo, classes, caminho):
    """Saída do primeiro bloco (Conv->BN->ReLU->Pool) para UMA imagem de teste."""
    with torch.no_grad():
        ativ = modelo.extrator[0](imagem.unsqueeze(0))[0]  # (canais, H, W)

    n = ativ.shape[0]
    colunas = 8
    linhas = (n + colunas - 1) // colunas
    fig, eixos = plt.subplots(linhas, colunas, figsize=(colunas * 1.3, linhas * 1.3))
    for i, ax in enumerate(eixos.ravel()):
        if i < n:
            ax.imshow(ativ[i].numpy(), cmap="viridis")
            ax.set_title(f"c{i}", fontsize=7)
        ax.axis("off")
    fig.suptitle(f"Mapas de ativação do bloco 1 — classe verdadeira: {classes[rotulo]}")
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)
    print(f"[ok] Mapas de ativação salvos em {caminho}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Visualização de filtros e ativações (3.4)")
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()

    dispositivo = obter_dispositivo()
    modelo, ckpt = carregar_checkpoint(dispositivo, args.tag)
    classes = ckpt["classes"]
    cfg = config_para_dataset(ckpt["config"]["dataset"])
    from dataclasses import replace
    cfg = replace(cfg, canais_conv=tuple(ckpt["config"]["canais_conv"]))

    out_dir = OUT_DIR / "filtros"
    out_dir.mkdir(parents=True, exist_ok=True)

    salvar_filtros(modelo, out_dir / f"filtros_conv1_{args.tag or 'baseline'}.png")

    _, _, carregador_teste, _ = obter_dataloaders(cfg)
    imagem, rotulo = carregador_teste.dataset[7]
    salvar_ativacoes(modelo, imagem, rotulo, classes,
                     out_dir / f"ativacoes_bloco1_{args.tag or 'baseline'}.png")
