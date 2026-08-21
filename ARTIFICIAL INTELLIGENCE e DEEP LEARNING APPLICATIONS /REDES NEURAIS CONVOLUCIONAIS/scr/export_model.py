"""
export_model.py — ETAPA 7-A: empacotar o modelo para produção
------------------------------------------------------------------
Mesma lógica de cnn/src/export_model.py, com --tag para escolher qual
checkpoint exportar (por padrão, o baseline em outputs/).

Uso:
    python export_model.py --tag baseline
"""

import argparse

import torch

from config import OUT_DIR
from evaluate import carregar_checkpoint
from utils import obter_dispositivo, salvar_json


def exportar_torchscript(modelo, classes, dispositivo, cfg, out_dir):
    modelo.eval()
    exemplo = torch.randn(1, cfg["canais_entrada"], cfg["tamanho_imagem"],
                          cfg["tamanho_imagem"], device=dispositivo)

    with torch.no_grad():
        modelo_scriptado = torch.jit.trace(modelo, exemplo)
    modelo_scriptado = torch.jit.freeze(modelo_scriptado)

    caminho = out_dir / "modelo_scriptado.pt"
    modelo_scriptado.save(str(caminho))
    salvar_json(classes, out_dir / "classes.json")

    recarregado = torch.jit.load(str(caminho), map_location=dispositivo)
    with torch.no_grad():
        original = modelo(exemplo)
        exportado = recarregado(exemplo)
    diferenca = (original - exportado).abs().max().item()

    print(f"[ok] TorchScript salvo em {caminho}")
    print(f"     diferença máxima original vs. exportado: {diferenca:.2e} "
          f"({'OK' if diferenca < 1e-4 else 'ATENÇÃO: divergência!'})")
    print(f"     tamanho do arquivo: {caminho.stat().st_size / 1024:.1f} KB")
    return caminho


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Exportação do modelo para produção")
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()

    dispositivo = obter_dispositivo()
    modelo, ckpt = carregar_checkpoint(dispositivo, args.tag)
    out_dir = OUT_DIR / args.tag if args.tag else OUT_DIR
    exportar_torchscript(modelo, ckpt["classes"], dispositivo, modelo.hiperparametros, out_dir)
