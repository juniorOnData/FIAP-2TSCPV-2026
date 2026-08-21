"""
evaluate.py — ETAPA 6 do pipeline: teste
-----------------------------------------
Extensão de cnn/src/evaluate.py:

  * --tag lê o checkpoint de outputs/<tag>/ (mesma pasta usada pelo train.py)
    em vez de sempre outputs/melhor_modelo.pt — necessário para avaliar cada
    um dos experimentos do Nível 2 isoladamente.
  * acuracia_topk() — exercício 3.3: acurácia top-2 (e genérica para top-k).
  * suporta checkpoints de MLPSimples e de datasets diferentes (CIFAR-10).
"""

import argparse

import torch
import torch.nn as nn

from config import CFG, OUT_DIR, config_para_dataset
from data import obter_dataloaders
from model import criar_modelo
from utils import obter_dispositivo, plotar_matriz_confusao, salvar_json


def carregar_checkpoint(dispositivo, tag: str = ""):
    out_dir = OUT_DIR / tag if tag else OUT_DIR
    caminho = out_dir / CFG.arq_checkpoint
    if not caminho.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado em {caminho}.")

    ckpt = torch.load(caminho, map_location=dispositivo, weights_only=False)
    tipo = ckpt.get("modelo_tipo", "cnn")
    modelo = criar_modelo(tipo, **ckpt["hiperparametros"]).to(dispositivo)
    modelo.load_state_dict(ckpt["model_state"])
    modelo.eval()
    return modelo, ckpt


@torch.no_grad()
def prever_conjunto(modelo, carregador, dispositivo, topk: int = 2):
    """Devolve (verdadeiros, previstos_top1, previstos_topk, perda_media)."""
    criterio = nn.CrossEntropyLoss(reduction="sum")
    verdadeiros, top1, topk_ind = [], [], []
    soma_perda, total = 0.0, 0

    for x, y in carregador:
        x, y = x.to(dispositivo), y.to(dispositivo)
        logits = modelo(x)
        soma_perda += criterio(logits, y).item()
        total += x.size(0)
        verdadeiros.append(y.cpu())
        top1.append(logits.argmax(dim=1).cpu())
        topk_ind.append(logits.topk(min(topk, logits.shape[1]), dim=1).indices.cpu())

    return (torch.cat(verdadeiros), torch.cat(top1), torch.cat(topk_ind),
            soma_perda / total)


def acuracia_topk(verdadeiros: torch.Tensor, topk_ind: torch.Tensor) -> float:
    """Fração de amostras cujo rótulo verdadeiro está entre as k mais prováveis."""
    acertos = (topk_ind == verdadeiros.unsqueeze(1)).any(dim=1)
    return acertos.float().mean().item()


def matriz_confusao(verdadeiros, previstos, n_classes: int):
    m = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for v, p in zip(verdadeiros, previstos):
        m[v, p] += 1
    return m


def metricas_por_classe(m, classes):
    resultado = {}
    for i, nome in enumerate(classes):
        vp = m[i, i].item()
        fp = (m[:, i].sum() - m[i, i]).item()
        fn = (m[i, :].sum() - m[i, i]).item()
        precisao = vp / (vp + fp) if vp + fp else 0.0
        revocacao = vp / (vp + fn) if vp + fn else 0.0
        f1 = 2 * precisao * revocacao / (precisao + revocacao) if precisao + revocacao else 0.0
        resultado[nome] = {"precisao": precisao, "revocacao": revocacao,
                           "f1": f1, "suporte": int(m[i, :].sum().item())}
    return resultado


def salvar_erros(modelo, carregador, classes, dispositivo, caminho, media, desvio, n: int = 12):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    from utils import _desnormalizar

    erros = []
    modelo.eval()
    with torch.no_grad():
        for x, y in carregador:
            logits = modelo(x.to(dispositivo))
            probs = torch.softmax(logits, dim=1).cpu()
            pred = probs.argmax(dim=1)
            conf = probs.max(dim=1).values
            for i in range(len(y)):
                if pred[i] != y[i]:
                    erros.append((conf[i].item(), x[i], y[i].item(), pred[i].item()))

    if not erros:
        return
    erros.sort(key=lambda e: -e[0])
    erros = erros[:n]

    linhas, colunas = 3, (n + 2) // 3
    fig, eixos = plt.subplots(linhas, colunas, figsize=(colunas * 1.9, linhas * 2.2))
    for ax, (conf, img, verdadeiro, previsto) in zip(eixos.ravel(), erros):
        arr = _desnormalizar(img, media, desvio)
        ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None)
        ax.set_title(f"real: {classes[verdadeiro]}\nprev: {classes[previsto]} ({conf:.0%})",
                     fontsize=7, color="darkred")
    for ax in eixos.ravel():
        ax.axis("off")
    fig.suptitle("Erros com maior confiança", fontsize=11)
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"[ok] Erros salvos em {caminho}")


def avaliar(tag: str = "", rapido: bool = False, sem_normalizacao: bool = False,
           sem_augmentation: bool = False) -> dict:
    """sem_normalizacao/sem_augmentation: PRECISAM bater com o que foi usado no
    treino deste checkpoint. transform_avaliacao nunca inclui augmentation, mas
    inclui (ou não) Normalize conforme essa flag — daí o mesmo erro clássico do
    deploy (paridade de pré-processamento) valer também aqui entre treino e
    avaliação."""
    dispositivo = obter_dispositivo()
    modelo, ckpt = carregar_checkpoint(dispositivo, tag)
    classes = ckpt["classes"]
    cfg = config_para_dataset(ckpt["config"]["dataset"])
    from dataclasses import replace
    cfg = replace(cfg, canais_conv=tuple(ckpt["config"]["canais_conv"]))

    print(f"[{tag or 'baseline'}] modelo da época {ckpt['epoca']} "
          f"(acc. validação {ckpt['val_acc'] * 100:.2f}%)\n")

    _, _, carregador_teste, _ = obter_dataloaders(
        cfg, rapido=rapido, sem_normalizacao=sem_normalizacao,
        sem_augmentation=sem_augmentation)
    verdadeiros, previstos, topk_ind, perda = prever_conjunto(modelo, carregador_teste, dispositivo)

    acuracia = (verdadeiros == previstos).float().mean().item()
    acc_top2 = acuracia_topk(verdadeiros, topk_ind)
    m = matriz_confusao(verdadeiros, previstos, len(classes))
    por_classe = metricas_por_classe(m, classes)

    print("=" * 66)
    print(f"TESTE — {len(verdadeiros)} imagens")
    print(f"Acurácia top-1: {acuracia * 100:.2f}%   |   Acurácia top-2: {acc_top2 * 100:.2f}%"
          f"   |   Perda: {perda:.4f}")
    print("=" * 66)
    print(f"{'classe':<16}{'precisão':>10}{'revocação':>12}{'F1':>8}{'n':>7}")
    for nome, met in por_classe.items():
        print(f"{nome:<16}{met['precisao']:>10.3f}{met['revocacao']:>12.3f}"
              f"{met['f1']:>8.3f}{met['suporte']:>7}")

    piores = sorted(por_classe.items(), key=lambda kv: kv[1]["f1"])[:3]
    print("\nClasses mais difíceis: " + ", ".join(f"{n} (F1={m_['f1']:.2f})" for n, m_ in piores))

    out_dir = OUT_DIR / tag if tag else OUT_DIR
    plotar_matriz_confusao(m.tolist(), classes, out_dir / cfg.arq_confusao)
    salvar_erros(modelo, carregador_teste, classes, dispositivo, out_dir / "erros_teste.png",
                cfg.media, cfg.desvio)

    resultado = {"acuracia_top1": acuracia, "acuracia_top2": acc_top2, "perda": perda,
                 "por_classe": por_classe, "matriz_confusao": m.tolist(),
                 "classes": classes, "tag": tag or "baseline"}
    salvar_json(resultado, out_dir / cfg.arq_metricas)
    salvar_json(classes, out_dir / cfg.arq_classes)
    return resultado


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Avaliação no conjunto de teste")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--rapido", action="store_true")
    p.add_argument("--sem-normalizacao", action="store_true")
    p.add_argument("--sem-augmentation", action="store_true")
    args = p.parse_args()
    avaliar(args.tag, args.rapido, args.sem_normalizacao, args.sem_augmentation)
