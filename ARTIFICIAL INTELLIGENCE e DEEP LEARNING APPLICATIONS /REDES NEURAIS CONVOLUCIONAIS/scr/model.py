"""
model.py — ETAPA 3 do pipeline: arquitetura
--------------------------------------------
Extensão de cnn/src/model.py:

  * bloco_conv(..., usar_bn=False) permite desligar o BatchNorm — exercício 2.6.
  * CNNSimples aceita canais_entrada != 1 e tamanho_imagem != 28, o que a torna
    reutilizável para o CIFAR-10 do Nível 4 (Caminho A) sem nenhuma mudança de
    código, só de configuração.
  * MLPSimples — exercício 3.1: o baseline denso que justifica a existência da
    convolução (achatar -> densa 128 -> ReLU -> densa 10).
"""

import torch
import torch.nn as nn

from config import CFG


def bloco_conv(entrada: int, saida: int, usar_bn: bool = True) -> nn.Sequential:
    """Bloco convolucional: Conv -> [BatchNorm] -> ReLU -> MaxPool.

    bias=True quando usar_bn=False: sem BatchNorm não há termo de deslocamento
    "de graça", então o bias da própria convolução volta a ser necessário.
    """
    camadas = [nn.Conv2d(entrada, saida, kernel_size=3, padding=1, bias=not usar_bn)]
    if usar_bn:
        camadas.append(nn.BatchNorm2d(saida))
    camadas += [nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*camadas)


class CNNSimples(nn.Module):
    """CNN de N blocos convolucionais + cabeça densa.

    Genérica em canais_entrada e tamanho_imagem: o mesmo código atende
    Fashion-MNIST (1x28x28) e CIFAR-10 (3x32x32, Nível 4 — Caminho A).
    """

    def __init__(self, n_classes: int = CFG.n_classes, canais_entrada: int = CFG.canais,
                 canais_conv: tuple = CFG.canais_conv, neuronios_fc: int = CFG.neuronios_fc,
                 dropout: float = CFG.dropout, tamanho_imagem: int = CFG.tamanho_imagem,
                 usar_batchnorm: bool = True):
        super().__init__()

        blocos = []
        c_ant = canais_entrada
        for c in canais_conv:
            blocos.append(bloco_conv(c_ant, c, usar_bn=usar_batchnorm))
            c_ant = c
        self.extrator = nn.Sequential(*blocos)

        with torch.no_grad():
            falso = torch.zeros(1, canais_entrada, tamanho_imagem, tamanho_imagem)
            n_achatado = self.extrator(falso).flatten(1).shape[1]

        self.classificador = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(n_achatado, neuronios_fc),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(neuronios_fc, n_classes),
        )

        self.hiperparametros = dict(
            n_classes=n_classes, canais_entrada=canais_entrada,
            canais_conv=tuple(canais_conv), neuronios_fc=neuronios_fc,
            dropout=dropout, tamanho_imagem=tamanho_imagem,
            usar_batchnorm=usar_batchnorm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extrator(x)
        return self.classificador(x)


class MLPSimples(nn.Module):
    """Baseline denso — exercício 3.1.

    Achata a imagem inteira e liga a uma única camada oculta de 128 neurônios.
    Não tem conectividade local nem pesos compartilhados: serve para medir
    quanto a convolução realmente compra em acurácia e em parâmetros.
    """

    def __init__(self, n_classes: int = CFG.n_classes, canais_entrada: int = CFG.canais,
                 tamanho_imagem: int = CFG.tamanho_imagem, neuronios_fc: int = 128):
        super().__init__()
        entrada = canais_entrada * tamanho_imagem * tamanho_imagem
        self.rede = nn.Sequential(
            nn.Flatten(),
            nn.Linear(entrada, neuronios_fc),
            nn.ReLU(inplace=True),
            nn.Linear(neuronios_fc, n_classes),
        )
        self.hiperparametros = dict(
            n_classes=n_classes, canais_entrada=canais_entrada,
            tamanho_imagem=tamanho_imagem, neuronios_fc=neuronios_fc,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rede(x)


def criar_modelo(tipo: str = "cnn", **kwargs):
    if tipo == "mlp":
        chaves = {"n_classes", "canais_entrada", "tamanho_imagem", "neuronios_fc"}
        return MLPSimples(**{k: v for k, v in kwargs.items() if k in chaves})
    return CNNSimples(**kwargs)


if __name__ == "__main__":
    from utils import contar_parametros

    for tipo in ("cnn", "mlp"):
        modelo = criar_modelo(tipo)
        print(f"\n=== {tipo.upper()} ===")
        print(modelo)
        print(f"Parâmetros treináveis: {contar_parametros(modelo):,}")

        x = torch.randn(4, CFG.canais, CFG.tamanho_imagem, CFG.tamanho_imagem)
        with torch.no_grad():
            y = modelo(x)
        print(f"Entrada {tuple(x.shape)} -> saída {tuple(y.shape)}")

        if tipo == "cnn":
            print("Formato após cada bloco convolucional:")
            z = x
            for i, bloco in enumerate(modelo.extrator, 1):
                with torch.no_grad():
                    z = bloco(z)
                print(f"  bloco {i}: {tuple(z.shape)}")
