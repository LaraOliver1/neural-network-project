"""
04_yolo.py — Objetivo extra: YOLO no modo classify (integrante 1)

Usa a MESMA divisão do grupo (split.json): monta as pastas no formato que o YOLO
espera, treina, escolhe o modelo pela validação e só olha o teste com --teste.

Saídas (mesmo padrão do common.py):
  modelos/yolo_cls.pt                 melhor modelo (escolhido na validação)
  historicos/yolo_cls.json            histórico por época + metadados (custo, F1 macro de validação)
  historicos/yolo_cls_epocas.csv      log bruto por época (results.csv do Ultralytics)
  figuras/curvas/yolo_cls.png         curvas de perda e acurácia (treino × validação)
  com --teste:
  resultados/yolo_cls_teste.json      métricas no teste (F1 macro, acurácia, por classe)
  figuras/yolo_cls_matriz_teste.png   matriz de confusão do teste (ordem fixa das classes)

Uso (Colab com GPU):
  pip install ultralytics
  python 04_yolo.py                 # treina e avalia na validação
  python 04_yolo.py --teste         # no FINAL: avalia o modelo salvo no teste (sem retreinar)
"""
import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np

SEED = 42
CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]  # ordem fixa do trabalho
NOME = "yolo_cls"

DIR_YOLO = Path("dados_yolo")
DIR_MODELOS = Path("modelos")
DIR_HISTORICOS = Path("historicos")
DIR_CURVAS = Path("figuras/curvas")
DIR_RESULTADOS = Path("resultados")


# ---------------------------------------------------------------------------
# 1. Pastas no formato do YOLO, a partir do split.json
# ---------------------------------------------------------------------------
def montar_pastas(split_path: str, data_dir: str):
    """dados_yolo/{train,val,test}/<classe>/<arquivo>.jpg — cópia fiel do split.json."""
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    if DIR_YOLO.exists():
        shutil.rmtree(DIR_YOLO)
    contagem = {}
    for conjunto in ["train", "val", "test"]:
        for c in CLASSES:
            (DIR_YOLO / conjunto / c).mkdir(parents=True, exist_ok=True)
        for e in split[conjunto]:
            origem = Path(data_dir) / e["path"]
            destino = DIR_YOLO / conjunto / e["label"] / Path(e["path"]).name
            shutil.copy2(origem, destino)
        contagem[conjunto] = len(split[conjunto])
    print(f"Pastas do YOLO montadas a partir de {split_path}: {contagem}")
    return split


def caminhos_e_rotulos(split, conjunto, data_dir):
    idx = {c: i for i, c in enumerate(CLASSES)}
    paths = [str(Path(data_dir) / e["path"]) for e in split[conjunto]]
    y = [idx[e["label"]] for e in split[conjunto]]
    return paths, np.array(y)


# ---------------------------------------------------------------------------
# 2. Predição na ordem FIXA das classes
#    (o YOLO ordena as pastas em ordem alfabética: cardboard, glass, metal...)
# ---------------------------------------------------------------------------
def prever(modelo, paths, imgsz, lote=64):
    nomes = modelo.names                                   # {indice_yolo: "classe"}
    para_fixo = {i: CLASSES.index(n) for i, n in nomes.items()}
    preds = []
    for i in range(0, len(paths), lote):
        for r in modelo.predict(paths[i:i + lote], imgsz=imgsz, verbose=False):
            preds.append(para_fixo[int(r.probs.top1)])
    return np.array(preds)


def metricas(y, p):
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    rel = classification_report(y, p, labels=list(range(len(CLASSES))), target_names=CLASSES,
                                output_dict=True, zero_division=0)
    return {
        "f1_macro": float(f1_score(y, p, average="macro")),
        "acuracia": float(accuracy_score(y, p)),
        "por_classe": {c: {k: round(float(rel[c][k]), 4) for k in ["precision", "recall", "f1-score", "support"]}
                       for c in CLASSES},
    }


# ---------------------------------------------------------------------------
# 3. Histórico e curvas
# ---------------------------------------------------------------------------
def ler_results_csv(caminho):
    with open(caminho, newline="") as f:
        linhas = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    num = lambda r, k: float(r[k]) if k in r and r[k] not in ("", None) else None
    return {
        "epoca": [int(float(r["epoch"])) for r in linhas],
        "loss": [num(r, "train/loss") for r in linhas],
        "val_loss": [num(r, "val/loss") for r in linhas],
        "val_accuracy": [num(r, "metrics/accuracy_top1") for r in linhas],
        "lr": [num(r, "lr/pg0") for r in linhas],
    }


def plotar_curvas(h, saida, melhor_epoca):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(h["epoca"], h["loss"], label="treino")
    a1.plot(h["epoca"], h["val_loss"], label="validação")
    a1.set_title("Perda"); a1.set_xlabel("Época"); a1.legend()
    a2.plot(h["epoca"], h["val_accuracy"], label="validação (top-1)", color="tab:orange")
    a2.set_title("Acurácia"); a2.set_xlabel("Época"); a2.set_ylim(0, 1); a2.legend()
    for a in (a1, a2):
        a.axvline(melhor_epoca, color="gray", ls="--", lw=1)
    fig.suptitle(f"{NOME} — linha tracejada: época do modelo salvo ({melhor_epoca})")
    fig.tight_layout()
    saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(saida, dpi=150)
    plt.close(fig)


def plotar_matriz(y, p, saida, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay.from_predictions(y, p, labels=list(range(len(CLASSES))), display_labels=CLASSES,
                                            cmap="Blues", colorbar=False, ax=ax)
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real"); ax.set_title(titulo)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(saida, dpi=150)
    plt.close(fig)


def hardware():
    import platform
    import torch
    gpu = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    return {"gpu": gpu, "cpu": platform.machine(), "torch": torch.__version__, "python": platform.python_version()}


# ---------------------------------------------------------------------------
# 4. Treino
# ---------------------------------------------------------------------------
def treinar(a, split):
    import torch
    import ultralytics
    from ultralytics import YOLO

    pesos_iniciais = a.modelo if a.pretreinado else a.modelo.replace(".pt", ".yaml")
    modelo = YOLO(pesos_iniciais)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    modelo.train(
        data=str(DIR_YOLO.resolve()), epochs=a.epocas, patience=a.paciencia, imgsz=a.imgsz,
        batch=a.batch, seed=SEED, deterministic=True, project=str(Path("runs").resolve()),
        name=NOME, exist_ok=True, workers=a.workers, plots=False, verbose=True,
    )
    tempo = time.time() - t0
    gpu_mb = round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None

    pasta = Path("runs") / NOME
    best = pasta / "weights" / "best.pt"
    DIR_MODELOS.mkdir(exist_ok=True)
    shutil.copy2(best, DIR_MODELOS / f"{NOME}.pt")

    DIR_HISTORICOS.mkdir(exist_ok=True)
    shutil.copy2(pasta / "results.csv", DIR_HISTORICOS / f"{NOME}_epocas.csv")
    h = ler_results_csv(pasta / "results.csv")

    # avaliação na VALIDAÇÃO, na ordem fixa das classes
    melhor = YOLO(str(DIR_MODELOS / f"{NOME}.pt"))
    X_val, y_val = caminhos_e_rotulos(split, "val", a.data_dir)
    m_val = metricas(y_val, prever(melhor, X_val, a.imgsz))

    # época do modelo salvo = a de maior acurácia top-1 de validação (critério do best.pt)
    accs = [v if v is not None else -1 for v in h["val_accuracy"]]
    epoca_salva = h["epoca"][int(np.argmax(accs))]

    n_params = sum(p.numel() for p in melhor.model.parameters())
    meta = {
        "experimento": NOME,
        "modelo": f"YOLO classify ({a.modelo})",
        "pretreinado": bool(a.pretreinado),
        "pretreinamento": "ImageNet (pesos do Ultralytics)" if a.pretreinado else "nenhum (treino do zero)",
        "imgsz": a.imgsz,
        "entrada": f"{a.imgsz} × {a.imgsz} (o YOLO classify redimensiona para quadrado)",
        "pesos_de_classe": None,
        "aumento_de_dados_treino": "padrão do Ultralytics para classify (só no treino)",
        "batch_size": a.batch,
        "epocas_max": a.epocas,
        "paciencia": a.paciencia,
        "epocas_rodadas": len(h["epoca"]),
        "epoca_do_modelo_salvo": epoca_salva,
        "criterio_modelo_salvo": "maior acurácia top-1 de validação (best.pt do Ultralytics)",
        "semente": SEED,
        "val_f1_macro": m_val["f1_macro"],
        "val_accuracy": m_val["acuracia"],
        "val_por_classe": m_val["por_classe"],
        "parametros_totais": int(n_params),
        "tamanho_modelo_salvo_mb": round((DIR_MODELOS / f"{NOME}.pt").stat().st_size / 2**20, 2),
        "tempo_total_treino_s": round(tempo, 1),
        "tempo_medio_por_epoca_s": round(tempo / max(len(h["epoca"]), 1), 2),
        "metodo_medicao_tempo": "time.time() antes e depois do model.train (treino + validação por época)",
        "memoria_gpu_pico_mb": gpu_mb,
        "hardware": hardware(),
        "ultralytics": ultralytics.__version__,
    }
    (DIR_HISTORICOS / f"{NOME}.json").write_text(
        json.dumps({"meta": meta, "historico": h}, indent=2, ensure_ascii=False), encoding="utf-8")
    plotar_curvas(h, DIR_CURVAS / f"{NOME}.png", epoca_salva)

    print("\nResumo (VALIDAÇÃO; o teste não foi usado):")
    print(f"  {NOME}: F1 macro={m_val['f1_macro']:.3f} | acurácia={m_val['acuracia']:.3f} | "
          f"épocas={len(h['epoca'])} (salvo {epoca_salva}) | {tempo / 60:.1f} min | "
          f"GPU pico={gpu_mb} MB | params={n_params:,}")
    print(f"  recall trash (val) = {m_val['por_classe']['trash']['recall']:.3f}")


# ---------------------------------------------------------------------------
# 5. Teste (só no final, sem retreinar nem ajustar nada depois)
# ---------------------------------------------------------------------------
def avaliar_teste(a, split):
    from ultralytics import YOLO
    caminho = DIR_MODELOS / f"{NOME}.pt"
    if not caminho.exists():
        raise SystemExit(f"Modelo não encontrado: {caminho}. Treine antes ou baixe do Drive.")
    meta_path = DIR_HISTORICOS / f"{NOME}.json"
    imgsz = json.loads(meta_path.read_text(encoding="utf-8"))["meta"]["imgsz"] if meta_path.exists() else a.imgsz

    modelo = YOLO(str(caminho))
    X, y = caminhos_e_rotulos(split, "test", a.data_dir)
    t0 = time.time()
    p = prever(modelo, X, imgsz)
    m = metricas(y, p)
    m["tempo_inferencia_teste_s"] = round(time.time() - t0, 1)
    m["n_imagens"] = len(y)

    DIR_RESULTADOS.mkdir(exist_ok=True)
    (DIR_RESULTADOS / f"{NOME}_teste.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    plotar_matriz(y, p, Path("figuras") / f"{NOME}_matriz_teste.png", f"{NOME} — teste")
    print(f"TESTE {NOME}: F1 macro={m['f1_macro']:.3f} | acurácia={m['acuracia']:.3f} | "
          f"recall trash={m['por_classe']['trash']['recall']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="split.json")
    ap.add_argument("--data_dir", default="dataset-resized")
    ap.add_argument("--modelo", default="yolo11n-cls.pt", help="ex.: yolo11n-cls.pt, yolo11s-cls.pt")
    ap.add_argument("--do_zero", dest="pretreinado", action="store_false",
                    help="treina sem pesos do ImageNet (comparação mais justa com MLP/CNN)")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--epocas", type=int, default=100)
    ap.add_argument("--paciencia", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--teste", action="store_true", help="só no FINAL: avalia o modelo salvo no teste")
    ap.add_argument("--forcar", action="store_true", help="treina de novo mesmo se o modelo já existir")
    a = ap.parse_args()

    split = json.loads(Path(a.split).read_text(encoding="utf-8"))
    if a.teste:
        avaliar_teste(a, split)
        return
    if (DIR_MODELOS / f"{NOME}.pt").exists() and not a.forcar:
        print(f"{NOME} já treinado (modelos/{NOME}.pt). Use --forcar para treinar de novo.")
        return
    montar_pastas(a.split, a.data_dir)
    treinar(a, split)


if __name__ == "__main__":
    main()
