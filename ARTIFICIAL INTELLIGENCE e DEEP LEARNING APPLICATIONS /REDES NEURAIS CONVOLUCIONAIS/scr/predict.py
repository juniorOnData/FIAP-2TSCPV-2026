"""
predict.py — inferência em imagens novas (exercício 3.5)
-----------------------------------------------------------
Idêntico em espírito a cnn/src/predict.py: TREINO e INFERÊNCIA precisam usar
exatamente o mesmo pré-processamento. Esta versão lê o dataset/config do
próprio checkpoint (via --tag), então funciona tanto para os modelos
Fashion-MNIST quanto para o modelo CIFAR-10 do Nível 4.

Uso:
    python predict.py foto.jpg --tag baseline
    python predict.py foto1.png foto2.jpg --tag baseline --inverter

Nota (exercício 3.5): este script depende de fotos reais fornecidas pelo
usuário — nenhuma foi fornecida nesta rodada, então o exercício 3.5 não foi
executado. O comando acima está pronto para uso assim que houver imagens.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from config import config_para_dataset
from evaluate import carregar_checkpoint
from utils import obter_dispositivo


def preparar_imagem(caminho: Path, cfg, inverter: bool = False) -> torch.Tensor:
    modo = "L" if cfg.canais == 1 else "RGB"
    imagem = Image.open(caminho).convert(modo)
    if inverter:
        from PIL import ImageOps
        imagem = ImageOps.invert(imagem) if modo == "L" else imagem

    transformacao = transforms.Compose([
        transforms.Resize((cfg.tamanho_imagem, cfg.tamanho_imagem)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.media, cfg.desvio),
    ])
    return transformacao(imagem).unsqueeze(0)


@torch.no_grad()
def prever(modelo, tensor, classes, dispositivo, k: int = 3):
    logits = modelo(tensor.to(dispositivo))
    probabilidades = torch.softmax(logits, dim=1)[0]
    valores, indices = probabilidades.topk(min(k, len(classes)))
    return [(classes[i], v.item()) for v, i in zip(valores, indices)]


def main():
    p = argparse.ArgumentParser(description="Classifica imagens com o modelo treinado")
    p.add_argument("imagens", nargs="+", type=Path)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--inverter", action="store_true")
    p.add_argument("--topk", type=int, default=3)
    args = p.parse_args()

    dispositivo = obter_dispositivo()
    modelo, ckpt = carregar_checkpoint(dispositivo, args.tag)
    classes = ckpt["classes"]
    cfg = config_para_dataset(ckpt["config"]["dataset"])
    cfg = replace(cfg, canais_conv=tuple(ckpt["config"]["canais_conv"]))

    for caminho in args.imagens:
        if not caminho.exists():
            print(f"[erro] arquivo não encontrado: {caminho}")
            continue
        tensor = preparar_imagem(caminho, cfg, args.inverter)
        resultado = prever(modelo, tensor, classes, dispositivo, args.topk)

        print(f"\n{caminho.name}")
        for posicao, (nome, prob) in enumerate(resultado, 1):
            barra = "#" * int(prob * 30)
            print(f"  {posicao}. {nome:<16} {prob * 100:6.2f}%  {barra}")

        if resultado[0][1] < 0.5:
            print("  [aviso] confiança baixa — imagem possivelmente fora do domínio")


if __name__ == "__main__":
    main()
