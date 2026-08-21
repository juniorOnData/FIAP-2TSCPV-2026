"""
train.py — ETAPAS 4 e 5 do pipeline: treinamento e validação
-------------------------------------------------------------
Extensão de cnn/src/train.py com os "botões" necessários para os experimentos
controlados do Nível 2 e o baseline do Nível 3.1, todos acionáveis por CLI
para que cada rodada fique registrada e reprodutível:

    --lr --dropout --canais-conv --sem-batchnorm --sem-normalizacao
    --sem-augmentation --modelo {cnn,mlp} --dataset {FashionMNIST,CIFAR10}
    --tag <pasta-de-saida>

Cada execução grava seus artefatos em outputs/<tag>/ (tag vazio = outputs/
direto, usado pelo baseline), nunca sobrescrevendo o experimento anterior.
"""

import argparse
import json
import time
from dataclasses import replace

import torch
import torch.nn as nn

from config import CFG, OUT_DIR, config_para_dataset, classes_do_dataset
from data import obter_dataloaders
from model import criar_modelo
from utils import (contar_parametros, definir_semente, obter_dispositivo,
                   plotar_curvas, salvar_json)


def executar_epoca(modelo, carregador, criterio, dispositivo, otimizador=None,
                   log_a_cada: int = 200, rotulo: str = ""):
    treinando = otimizador is not None
    modelo.train() if treinando else modelo.eval()

    soma_perda, acertos, total = 0.0, 0, 0
    contexto = torch.enable_grad() if treinando else torch.no_grad()

    with contexto:
        for i, (x, y) in enumerate(carregador, start=1):
            x, y = x.to(dispositivo), y.to(dispositivo)

            logits = modelo(x)
            perda = criterio(logits, y)

            if treinando:
                otimizador.zero_grad(set_to_none=True)
                perda.backward()
                otimizador.step()

            soma_perda += perda.item() * x.size(0)
            acertos += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)

            if treinando and i % log_a_cada == 0:
                print(f"    {rotulo} lote {i:>4}/{len(carregador)} | "
                      f"perda {soma_perda / total:.4f} | acc {acertos / total:.3f}")

    return soma_perda / total, acertos / total


def treinar(args) -> dict:
    definir_semente(CFG.semente)
    dispositivo = obter_dispositivo()

    cfg = config_para_dataset(args.dataset)
    overrides = {}
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.dropout is not None:
        overrides["dropout"] = args.dropout
    if args.canais_conv is not None:
        overrides["canais_conv"] = tuple(int(c) for c in args.canais_conv.split(","))
    if args.epocas is not None:
        overrides["epocas"] = args.epocas
    overrides["usar_batchnorm"] = not args.sem_batchnorm
    cfg = replace(cfg, **overrides)

    classes = classes_do_dataset(cfg.dataset)
    out_dir = OUT_DIR / args.tag if args.tag else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Experimento: {args.tag or 'baseline'} ===")
    print(f"Dispositivo: {dispositivo} | dataset: {cfg.dataset} | modelo: {args.modelo}")
    print(f"lr={cfg.lr} dropout={cfg.dropout} canais_conv={cfg.canais_conv} "
          f"batchnorm={cfg.usar_batchnorm} normalizacao={not args.sem_normalizacao} "
          f"augmentation={not args.sem_augmentation}\n")

    carregador_treino, carregador_val, carregador_teste, _ = obter_dataloaders(
        cfg, rapido=args.rapido, sem_normalizacao=args.sem_normalizacao,
        sem_augmentation=args.sem_augmentation)
    print(f"Treino: {len(carregador_treino.dataset)} | Validação: {len(carregador_val.dataset)}")

    modelo = criar_modelo(
        args.modelo, n_classes=cfg.n_classes, canais_entrada=cfg.canais,
        canais_conv=cfg.canais_conv, neuronios_fc=cfg.neuronios_fc,
        dropout=cfg.dropout, tamanho_imagem=cfg.tamanho_imagem,
        usar_batchnorm=cfg.usar_batchnorm,
    ).to(dispositivo)
    n_params = contar_parametros(modelo)
    print(f"Parâmetros treináveis: {n_params:,}\n")

    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    escalonador = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, mode="max", factor=0.5, patience=1)

    historico = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
                 "lr": [], "tempo_epoca_s": []}
    melhor_acc, melhor_epoca, epocas_sem_melhora = 0.0, 0, 0
    caminho_ckpt = out_dir / cfg.arq_checkpoint
    t_inicio = time.time()

    for epoca in range(1, cfg.epocas + 1):
        t0 = time.time()
        print(f"Época {epoca}/{cfg.epocas}")

        perda_tr, acc_tr = executar_epoca(modelo, carregador_treino, criterio,
                                          dispositivo, otimizador, rotulo="treino")
        perda_va, acc_va = executar_epoca(modelo, carregador_val, criterio, dispositivo)

        lr_atual = otimizador.param_groups[0]["lr"]
        escalonador.step(acc_va)
        dt = time.time() - t0

        historico["train_loss"].append(perda_tr)
        historico["train_acc"].append(acc_tr)
        historico["val_loss"].append(perda_va)
        historico["val_acc"].append(acc_va)
        historico["lr"].append(lr_atual)
        historico["tempo_epoca_s"].append(dt)

        print(f"  treino   -> perda {perda_tr:.4f} | acc {acc_tr * 100:.2f}%")
        print(f"  validação-> perda {perda_va:.4f} | acc {acc_va * 100:.2f}%"
              f"  (lr={lr_atual:.1e}, {dt:.1f}s)")

        if acc_va > melhor_acc:
            melhor_acc, melhor_epoca, epocas_sem_melhora = acc_va, epoca, 0
            torch.save({
                "model_state": modelo.state_dict(),
                "hiperparametros": modelo.hiperparametros,
                "modelo_tipo": args.modelo,
                "classes": classes,
                "epoca": epoca,
                "val_acc": acc_va,
                "config": cfg.to_dict(),
            }, caminho_ckpt)
            print(f"  [ok] novo melhor modelo salvo ({acc_va * 100:.2f}%)")
        else:
            epocas_sem_melhora += 1
            print(f"  sem melhora há {epocas_sem_melhora} época(s)")

        if epocas_sem_melhora >= cfg.paciencia:
            print(f"\nEarly stopping na época {epoca} (paciência = {cfg.paciencia}).")
            break
        print()

    tempo_total = time.time() - t_inicio
    tempo_medio_epoca = sum(historico["tempo_epoca_s"]) / len(historico["tempo_epoca_s"])

    print(f"\nMelhor acurácia de validação: {melhor_acc * 100:.2f}% (época {melhor_epoca})")
    print(f"Tempo total: {tempo_total:.1f}s | Tempo médio/época: {tempo_medio_epoca:.1f}s")
    print(f"Checkpoint: {caminho_ckpt}")

    resultado = {"historico": historico, "melhor_val_acc": melhor_acc,
                 "melhor_epoca": melhor_epoca, "n_parametros": n_params,
                 "tempo_total_s": tempo_total, "tempo_medio_epoca_s": tempo_medio_epoca,
                 "config": cfg.to_dict(), "modelo_tipo": args.modelo,
                 "sem_normalizacao": args.sem_normalizacao,
                 "sem_augmentation": args.sem_augmentation, "tag": args.tag or "baseline"}
    salvar_json(resultado, out_dir / cfg.arq_historico)
    plotar_curvas(historico, out_dir / cfg.arq_curvas)
    return resultado


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Treinamento da CNN / MLP")
    p.add_argument("--epocas", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--canais-conv", type=str, default=None, help='ex.: "8,16,32"')
    p.add_argument("--sem-batchnorm", action="store_true")
    p.add_argument("--sem-normalizacao", action="store_true")
    p.add_argument("--sem-augmentation", action="store_true")
    p.add_argument("--modelo", choices=["cnn", "mlp"], default="cnn")
    p.add_argument("--dataset", choices=["FashionMNIST", "CIFAR10"], default="FashionMNIST")
    p.add_argument("--tag", type=str, default="", help="subpasta em outputs/ para este experimento")
    p.add_argument("--rapido", action="store_true")
    treinar(p.parse_args())
