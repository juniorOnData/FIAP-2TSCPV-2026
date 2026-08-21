"""
experimentos.py — driver dos exercícios de Nível 2 e do baseline do 3.1
--------------------------------------------------------------------------
Roda, em sequência, o baseline + o MLP (3.1) + as 12 ablações do Nível 2
(uma variável por vez, sempre contra a mesma linha de base), avalia cada
checkpoint no conjunto de TESTE (mexer nisso é o que a rubrica pune) e
consolida tudo em outputs/experimentos/resumo.json — a fonte dos dados do
relatório final.

Uso:  python experimentos.py
"""

import time
import traceback
from types import SimpleNamespace

from config import OUT_DIR
from train import treinar
from evaluate import avaliar
from utils import salvar_json

DEFAULTS = dict(lr=None, dropout=None, canais_conv=None, epocas=None,
                sem_batchnorm=False, sem_normalizacao=False, sem_augmentation=False,
                modelo="cnn", dataset="FashionMNIST", rapido=False)


def args_de(tag: str, **overrides) -> SimpleNamespace:
    d = dict(DEFAULTS)
    d.update(overrides)
    d["tag"] = tag
    return SimpleNamespace(**d)


EXPERIMENTOS = [
    ("baseline", args_de("baseline", epocas=12)),
    ("3.1 MLP baseline", args_de("mlp_baseline", epocas=12, modelo="mlp")),
    ("2.1 sem normalizacao", args_de("2_1_sem_normalizacao", epocas=5, sem_normalizacao=True)),
    ("2.2 sem augmentation", args_de("2_2_sem_augmentation", epocas=5, sem_augmentation=True)),
    ("2.3 canais (8,16,32)", args_de("2_3_canais_pequeno", epocas=5, canais_conv="8,16,32")),
    ("2.3 canais (32,64,128)", args_de("2_3_canais_grande", epocas=5, canais_conv="32,64,128")),
    ("2.4 lr=1e-1", args_de("2_4_lr_1e-1", epocas=5, lr=1e-1)),
    ("2.4 lr=1e-3", args_de("2_4_lr_1e-3", epocas=5, lr=1e-3)),
    ("2.4 lr=1e-5", args_de("2_4_lr_1e-5", epocas=5, lr=1e-5)),
    ("2.5 dropout=0.0", args_de("2_5_dropout_0.0", epocas=5, dropout=0.0)),
    ("2.5 dropout=0.3", args_de("2_5_dropout_0.3", epocas=5, dropout=0.3)),
    ("2.5 dropout=0.7", args_de("2_5_dropout_0.7", epocas=5, dropout=0.7)),
    ("2.6 sem BatchNorm, lr=1e-3", args_de("2_6_sembn_lr1e-3", epocas=5, sem_batchnorm=True, lr=1e-3)),
    ("2.6 sem BatchNorm, lr=1e-2", args_de("2_6_sembn_lr1e-2", epocas=5, sem_batchnorm=True, lr=1e-2)),
]


def main():
    resumo = []
    t_geral = time.time()

    for i, (descricao, args) in enumerate(EXPERIMENTOS, start=1):
        print(f"\n{'#'*70}\n# [{i}/{len(EXPERIMENTOS)}] {descricao}  (tag={args.tag})\n{'#'*70}")
        try:
            r_treino = treinar(args)
            r_teste = avaliar(tag=args.tag, rapido=False,
                              sem_normalizacao=args.sem_normalizacao,
                              sem_augmentation=args.sem_augmentation)
            resumo.append({
                "descricao": descricao,
                "tag": args.tag,
                "val_acc": r_treino["melhor_val_acc"],
                "teste_acc_top1": r_teste["acuracia_top1"],
                "teste_acc_top2": r_teste["acuracia_top2"],
                "n_parametros": r_treino["n_parametros"],
                "tempo_medio_epoca_s": r_treino["tempo_medio_epoca_s"],
                "tempo_total_s": r_treino["tempo_total_s"],
                "epocas_rodadas": len(r_treino["historico"]["train_loss"]),
                "config": r_treino["config"],
                "modelo_tipo": args.modelo,
                "ok": True,
            })
        except Exception as e:
            print(f"[ERRO] experimento {descricao} falhou: {e}")
            traceback.print_exc()
            resumo.append({"descricao": descricao, "tag": args.tag, "ok": False,
                           "erro": str(e)})

        salvar_json(resumo, OUT_DIR / "experimentos" / "resumo.json")

    print(f"\nTodos os experimentos concluídos em {(time.time()-t_geral)/60:.1f} min.")
    print(f"Resumo em {OUT_DIR / 'experimentos' / 'resumo.json'}")


if __name__ == "__main__":
    main()
