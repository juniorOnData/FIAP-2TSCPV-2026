"""
utils.py
--------
Funções de apoio: semente aleatória, dispositivo, JSON, gráficos.

Extensão de cnn/src/utils.py: as funções de plot passaram a aceitar imagens
com 1 ou 3 canais (necessário para o CIFAR-10 do Nível 4 — Caminho A).
"""

import json
import random
from pathlib import Path

import numpy as np
import torch


def definir_semente(semente: int = 42) -> None:
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)


def obter_dispositivo() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def salvar_json(dados: dict, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=2, ensure_ascii=False)


def carregar_json(caminho: Path) -> dict:
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


def contar_parametros(modelo: torch.nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def _desnormalizar(img: torch.Tensor, media: tuple, desvio: tuple) -> np.ndarray:
    """img: (C,H,W) normalizada -> array (H,W) ou (H,W,C) em [0,1], pronto p/ imshow."""
    media_t = torch.tensor(media).view(-1, 1, 1)
    desvio_t = torch.tensor(desvio).view(-1, 1, 1)
    img = (img * desvio_t + media_t).clamp(0, 1)
    if img.shape[0] == 1:
        return img.squeeze(0).numpy()
    return img.permute(1, 2, 0).numpy()  # (C,H,W) -> (H,W,C)


def plotar_curvas(historico: dict, caminho: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[aviso] matplotlib não instalado; pulando gráficos.")
        return

    epocas = range(1, len(historico["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epocas, historico["train_loss"], "o-", label="treino")
    ax1.plot(epocas, historico["val_loss"], "s-", label="validação")
    ax1.set_xlabel("época"); ax1.set_ylabel("perda (cross-entropy)")
    ax1.set_title("Curva de perda"); ax1.legend(); ax1.grid(alpha=.3)

    ax2.plot(epocas, [a * 100 for a in historico["train_acc"]], "o-", label="treino")
    ax2.plot(epocas, [a * 100 for a in historico["val_acc"]], "s-", label="validação")
    ax2.set_xlabel("época"); ax2.set_ylabel("acurácia (%)")
    ax2.set_title("Curva de acurácia"); ax2.legend(); ax2.grid(alpha=.3)

    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"[ok] Curvas salvas em {caminho}")


def plotar_matriz_confusao(matriz, classes, caminho: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[aviso] matplotlib não instalado; pulando matriz de confusão.")
        return

    m = np.array(matriz, dtype=float)
    linhas = m.sum(axis=1, keepdims=True)
    linhas[linhas == 0] = 1
    m_norm = m / linhas

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(m_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Classe prevista"); ax.set_ylabel("Classe verdadeira")
    ax.set_title("Matriz de confusão (normalizada por linha)")

    for i in range(len(classes)):
        for j in range(len(classes)):
            if m_norm[i, j] > 0.005:
                ax.text(j, i, f"{m_norm[i, j]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if m_norm[i, j] > 0.5 else "black")

    fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"[ok] Matriz de confusão salva em {caminho}")


def salvar_amostra(carregador, classes, caminho: Path, media: tuple, desvio: tuple,
                   n: int = 16) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[aviso] matplotlib não instalado; pulando amostra.")
        return

    imagens, rotulos = next(iter(carregador))
    imagens = imagens[:n]

    linhas = int(n ** 0.5)
    colunas = (n + linhas - 1) // linhas
    fig, eixos = plt.subplots(linhas, colunas, figsize=(colunas * 1.6, linhas * 1.8))
    for i, ax in enumerate(eixos.ravel()):
        if i < n:
            img = _desnormalizar(imagens[i], media, desvio)
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(classes[rotulos[i]], fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"[ok] Amostra salva em {caminho}")
