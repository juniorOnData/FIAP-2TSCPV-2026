"""
config.py
---------
Ponto único de configuração do projeto (extensão do config.py da aula).

Diferenças em relação a cnn/src/config.py:
  * media/desvio viram tuplas, para suportar tanto imagens em escala de
    cinza (1 canal, Fashion-MNIST) quanto coloridas (3 canais, CIFAR-10 —
    Nível 4, Caminho A).
  * usar_batchnorm foi adicionado para a ablação do exercício 2.6.
  * dataset passa a aceitar "FashionMNIST" ou "CIFAR10".
"""

from dataclasses import dataclass, asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

# Estatísticas de normalização por dataset (calculadas sobre o TREINO).
ESTATISTICAS = {
    "FashionMNIST": {"media": (0.2860,), "desvio": (0.3530,), "canais": 1, "tamanho": 28},
    "CIFAR10": {"media": (0.4914, 0.4822, 0.4465), "desvio": (0.2470, 0.2435, 0.2616),
                "canais": 3, "tamanho": 32},
}

CLASSES_FASHION = [
    "Camiseta/Top", "Calça", "Pulôver", "Vestido", "Casaco",
    "Sandália", "Camisa", "Tênis", "Bolsa", "Bota",
]
CLASSES_CIFAR10 = [
    "Avião", "Automóvel", "Pássaro", "Gato", "Veado",
    "Cachorro", "Sapo", "Cavalo", "Navio", "Caminhão",
]


@dataclass
class Config:
    # --- Dados ---------------------------------------------------------
    dataset: str = "FashionMNIST"
    tamanho_imagem: int = 28
    canais: int = 1
    n_classes: int = 10
    frac_validacao: float = 0.1
    num_workers: int = 0

    media: tuple = (0.2860,)
    desvio: tuple = (0.3530,)

    # --- Treinamento ---------------------------------------------------
    batch_size: int = 128
    epocas: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    paciencia: int = 3
    semente: int = 42

    # --- Arquitetura -----------------------------------------------------
    canais_conv: tuple = (16, 32, 64)
    neuronios_fc: int = 128
    dropout: float = 0.3
    usar_batchnorm: bool = True

    # --- Arquivos de saída (nomes-base; o caminho completo ganha o
    #     prefixo outputs/<tag>/ definido em cada experimento) -----------
    arq_checkpoint: str = "melhor_modelo.pt"
    arq_historico: str = "historico.json"
    arq_curvas: str = "curvas_treino.png"
    arq_confusao: str = "matriz_confusao.png"
    arq_metricas: str = "metricas_teste.json"
    arq_torchscript: str = "modelo_scriptado.pt"
    arq_classes: str = "classes.json"

    def to_dict(self) -> dict:
        return asdict(self)


def config_para_dataset(dataset: str, base: "Config" = None) -> Config:
    """Devolve uma cópia de `base` com canais/tamanho/estatísticas do dataset."""
    base = base or Config()
    info = ESTATISTICAS[dataset]
    return replace(base, dataset=dataset, canais=info["canais"],
                   tamanho_imagem=info["tamanho"], media=info["media"],
                   desvio=info["desvio"])


def classes_do_dataset(dataset: str) -> list:
    return CLASSES_FASHION if dataset == "FashionMNIST" else CLASSES_CIFAR10


CFG = Config()
CLASSES = CLASSES_FASHION  # retrocompatibilidade com scripts que importam CLASSES direto
