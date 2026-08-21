"""
api.py — ETAPA 7-B: servir o modelo baseline como uma API web
------------------------------------------------------------------
Cópia adaptada de cnn/deploy/api.py: mesma filosofia (não importa nada de
src/, carrega o modelo uma única vez no lifespan, reimplementa o
pré-processamento sem torchvision). Serve o checkpoint TorchScript do
baseline (outputs/modelo_scriptado.pt), gerado por src/export_model.py.

Como rodar (a partir da pasta cnn_junior/):
    python -m uvicorn deploy.api:app --reload --port 8000
"""

import io
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, ImageOps

TAMANHO = 28
MEDIA = 0.2860
DESVIO = 0.3530

RAIZ = Path(__file__).resolve().parents[1]
CAMINHO_MODELO = RAIZ / "outputs" / "modelo_scriptado.pt"
CAMINHO_CLASSES = RAIZ / "outputs" / "classes.json"

modelo = None
classes: list[str] = []


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    global modelo, classes
    if not CAMINHO_MODELO.exists():
        raise RuntimeError(
            f"Modelo não encontrado em {CAMINHO_MODELO}.\n"
            "Rode antes:  python src/train.py --tag baseline --epocas 12  e  "
            "python src/export_model.py --tag baseline")
    modelo = torch.jit.load(str(CAMINHO_MODELO), map_location="cpu")
    modelo.eval()
    classes = json.loads(CAMINHO_CLASSES.read_text(encoding="utf-8"))
    print(f"[startup] modelo carregado | {len(classes)} classes")
    yield
    print("[shutdown] encerrando serviço")


app = FastAPI(
    title="Classificador de Roupas (CNN) — cnn_junior",
    description="Serviço de inferência da CNN treinada no FashionMNIST",
    version="1.0.0",
    lifespan=ciclo_de_vida,
)


def preprocessar(bytes_imagem: bytes, inverter: bool = False) -> torch.Tensor:
    imagem = Image.open(io.BytesIO(bytes_imagem)).convert("L")
    if inverter:
        imagem = ImageOps.invert(imagem)
    imagem = imagem.resize((TAMANHO, TAMANHO))

    bruto = torch.frombuffer(bytearray(imagem.tobytes()), dtype=torch.uint8)
    tensor = bruto.float().reshape(1, 1, TAMANHO, TAMANHO) / 255.0
    return (tensor - MEDIA) / DESVIO


@app.get("/saude")
def saude():
    return {"status": "ok", "modelo_carregado": modelo is not None,
            "n_classes": len(classes)}


@app.post("/prever")
async def prever(arquivo: UploadFile = File(...), inverter: bool = False):
    if not arquivo.content_type or not arquivo.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie um arquivo de imagem.")

    conteudo = await arquivo.read()
    try:
        tensor = preprocessar(conteudo, inverter)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Imagem inválida: {e}")

    t0 = time.perf_counter()
    with torch.no_grad():
        probabilidades = torch.softmax(modelo(tensor), dim=1)[0]
    ms = (time.perf_counter() - t0) * 1000

    valores, indices = probabilidades.topk(3)
    return {
        "arquivo": arquivo.filename,
        "predicao": classes[indices[0]],
        "confianca": round(valores[0].item(), 4),
        "top3": [{"classe": classes[i], "probabilidade": round(v.item(), 4)}
                 for v, i in zip(valores, indices)],
        "tempo_inferencia_ms": round(ms, 2),
    }


@app.get("/", response_class=HTMLResponse)
def pagina_teste():
    return """
<!doctype html><html lang="pt-br"><meta charset="utf-8">
<title>Classificador de Roupas — cnn_junior</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:640px;margin:3rem auto;padding:0 1rem}
 #saida{white-space:pre-wrap;background:#f4f4f5;padding:1rem;border-radius:8px;margin-top:1rem}
 button{padding:.6rem 1.2rem;border:0;border-radius:6px;background:#2563eb;color:#fff;cursor:pointer}
</style>
<h1>Classificador de Roupas — CNN (cnn_junior)</h1>
<p>Envie uma imagem (idealmente uma peça de roupa clara sobre fundo escuro).</p>
<input type="file" id="arq" accept="image/*">
<label><input type="checkbox" id="inv"> inverter cores</label>
<button onclick="enviar()">Classificar</button>
<div id="saida">aguardando…</div>
<script>
async function enviar(){
  const f = document.getElementById('arq').files[0];
  if(!f){ document.getElementById('saida').textContent = 'Selecione um arquivo.'; return; }
  const fd = new FormData(); fd.append('arquivo', f);
  const inv = document.getElementById('inv').checked;
  document.getElementById('saida').textContent = 'processando…';
  const r = await fetch('/prever?inverter=' + inv, {method:'POST', body: fd});
  document.getElementById('saida').textContent = JSON.stringify(await r.json(), null, 2);
}
</script></html>
"""
