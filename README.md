# TrashNet — MLP × CNN (Redes Neurais Artificiais)

Comparação entre MLP e CNN na classificação de 2.527 imagens do TrashNet (6 classes), avaliando o efeito dos **pesos de classe** e da **resolução**.

> **Status:** ✅ Dados e EDA concluídos (`split.json` + figuras). Próximo passo: treinar os 6 experimentos.

---

## 1. Antes de tudo: regras que valem pra todo mundo

- **Ninguém refaz a divisão.** Todos os experimentos carregam `split.json`. Nada de `train_test_split` no seu script.
- **Pesos de classe e aumento de dados: só no treino.**
- **A validação escolhe as configurações** (épocas, taxa de aprendizado, arquitetura).
- **O teste só é usado no final.** Não olhe o teste para ajustar nada.
- **Ordem das classes fixa** (inclusive nas matrizes de confusão):
  `glass, paper, cardboard, plastic, metal, trash` → índices `0, 1, 2, 3, 4, 5`

---

## 2. Baixar o dataset

As imagens **não** estão neste repositório. Baixe do repositório oficial e extraia na raiz do projeto:

**Colab / Linux**
```bash
git clone https://github.com/garythung/trashnet.git
unzip -q trashnet/data/dataset-resized.zip -d .
```

**Jupyter no Windows**
```python
!git clone https://github.com/garythung/trashnet.git
import zipfile
zipfile.ZipFile("trashnet/data/dataset-resized.zip").extractall(".")
```

Estrutura esperada:
```
dataset-resized/
├── cardboard/  glass/  metal/  paper/  plastic/  trash/
```

---

## 3. Conteúdo do repositório (parte de dados)

| Arquivo | O que é |
|---|---|
| `01_eda_split.py` | Confere integridade, conta classes, faz o split 70/15/15 (seed 42), calcula pesos de classe e gera as figuras |
| `split.json` | **A divisão oficial.** Lista de imagens de treino/validação/teste + pesos de classe |
| `saida_eda.txt` | Log da execução (contagens, duplicatas, pesos) |
| `figuras/eda/` | Distribuição de classes, distribuição por conjunto, amostras por classe, cor média por classe, mesma imagem nas 3 resoluções, duplicatas |
| `common.py` | Código compartilhado pelos 6 treinos: semente, classes, pipeline de dados, callbacks e registro de custo |
| `02_mlp.py` | Treino dos experimentos `mlp_128_sem` e `mlp_128_com` |
| `treino_mlp_colab.ipynb` | Notebook para rodar o `02_mlp.py` no Colab |
| `03_cnn.py` | Treino dos experimentos CNN: `cnn_128_sem` e `cnn_128_com` (padrão); também `cnn_256_sem` e `cnn_512_sem` (pessoa 4), com a **mesma** arquitetura |
| `treino_cnn_colab.ipynb` | Notebook para rodar o `03_cnn.py` no Colab + comparação MLP × CNN e efeito dos pesos por classe (só validação) |

**Pessoa 4 (resolução):** use o mesmo script, que garante a mesma CNN (com global average pooling, os parâmetros são iguais nas 3 resoluções):
```bash
python 03_cnn.py --exp cnn_256_sem cnn_512_sem
python 03_cnn.py --exp cnn_512_sem --batch_size 16   # só se faltar memória (registre e justifique)
```

Para regenerar (não é necessário, o resultado é sempre o mesmo):
```bash
python 01_eda_split.py --data_dir dataset-resized
```

---

## 4. Números do dataset

| Classe | Total | Treino | Validação | Teste | Peso de classe |
|---|---|---|---|---|---|
| glass | 501 | 351 | 75 | 75 | 0.84 |
| paper | 594 | 415 | 89 | 90 | 0.7104 |
| cardboard | 403 | 282 | 61 | 60 | 1.0455 |
| plastic | 482 | 338 | 72 | 72 | 0.8723 |
| metal | 410 | 287 | 62 | 61 | 1.0273 |
| trash | 137 | 96 | 20 | 21 | 3.0712 |
| **Total** | **2527** | **1769** | **379** | **379** | |

- Todas as imagens: 512×384, nenhuma corrompida.
- Desbalanceamento: trash = 5,4% do total; razão maior/menor classe = 4,34.
- Pesos calculados **só no treino** com `compute_class_weight("balanced")`.
- ⚠️ **3 imagens duplicadas com rótulos diferentes** (mantidas, citadas como limitação no artigo):
  `glass115 = metal91`, `glass176 = plastic152`, `glass389 = plastic332`.
  As 6 cópias ficam **todas no treino**, para que nenhuma imagem apareça em dois conjuntos
  (sem vazamento para validação/teste). O script confere isso pelo hash de cada arquivo.
  Ver `figuras/eda/duplicatas.png`.

---

## 5. Como usar o `split.json` no seu treino

Formato:
```json
{
  "seed": 42,
  "classes": ["glass", "paper", "cardboard", "plastic", "metal", "trash"],
  "class_weights": {"glass": 0.8419, "...": "..."},
  "train": [{"path": "glass/glass1.jpg", "label": "glass"}, ...],
  "val":   [...],
  "test":  [...]
}
```
Os caminhos são **relativos à pasta `dataset-resized/`**.

### Carregar a divisão (qualquer framework)
```python
import json
from pathlib import Path

DATA_DIR = Path("dataset-resized")
split = json.load(open("split.json"))
CLASSES = split["classes"]
IDX = {c: i for i, c in enumerate(CLASSES)}

def carregar(conjunto):
    paths  = [str(DATA_DIR / e["path"]) for e in split[conjunto]]
    labels = [IDX[e["label"]] for e in split[conjunto]]
    return paths, labels

X_train, y_train = carregar("train")
X_val,   y_val   = carregar("val")
X_test,  y_test  = carregar("test")   # só no final!

# pesos de classe no formato {índice: peso} (Keras usa assim)
class_weight = {IDX[c]: w for c, w in split["class_weights"].items()}
```

### Exemplo com Keras / tf.data
```python
import tensorflow as tf

IMG_H, IMG_W = 96, 128   # 128×96 → (altura, largura) = (96, 128)
                         # 256×192 → (192, 256) | 512×384 → (384, 512)

def ler(path, label):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (IMG_H, IMG_W)) / 255.0
    return img, label

def dataset(paths, labels, treino=False, batch=32):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if treino:
        ds = ds.shuffle(len(paths), seed=42)
    return ds.map(ler, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)

train_ds = dataset(X_train, y_train, treino=True)
val_ds   = dataset(X_val, y_val)

# experimento "_com": passe class_weight | experimento "_sem": não passe
model.fit(train_ds, validation_data=val_ds, epochs=..., class_weight=class_weight)
```

> **Atenção à resolução:** o trabalho fala em largura×altura (128×96), mas `tf.image.resize` e `torchvision.transforms.Resize` recebem **(altura, largura)**.

---

## 6. Divisão do trabalho

| # | Responsável | Tarefa | Entrega |
|---|---|---|---|
| 1 | Dados e EDA | `01_eda_split.py`, extra YOLO (`04_yolo.py`) | ✅ `split.json`, figuras · ⏳ YOLO |
| 2 | MLP | `mlp_128_sem`, `mlp_128_com` | 2 modelos, histórico, texto |
| 3 | CNN 128 | `03_cnn.py`: `cnn_128_sem`, `cnn_128_com` | ✅ script · ⏳ 2 modelos, histórico, texto |
| 4 | Resolução | `cnn_256_sem`, `cnn_512_sem` + tempo e memória | 2 modelos, custo × ganho |
| 5 | Resultados | `05_resultados.py`, tabelas, matrizes, curvas, análise de erros | Seção de resultados |
| 6 | README e revisão | README final, modelos no Drive, ABNT, limite de 10 páginas | Repositório e PDF final |

### Os 6 experimentos

| Experimento | Modelo | Resolução | Pesos de classe |
|---|---|---|---|
| `mlp_128_sem` | MLP | 128×96 | não |
| `mlp_128_com` | MLP | 128×96 | sim |
| `cnn_128_sem` | CNN | 128×96 | não |
| `cnn_128_com` | CNN | 128×96 | sim |
| `cnn_256_sem` | CNN | 256×192 | não |
| `cnn_512_sem` | CNN | 512×384 | não |

**Para quem treina (pessoas 2, 3 e 4):** salve o modelo e o histórico com o nome do experimento (ex.: `modelos/cnn_128_com.keras`, `historicos/cnn_128_com.json`). A pessoa 5 vai precisar disso e a análise de erros usa o modelo com **maior F1 macro na validação**.

---

## 7. Prazos

| Data | Meta |
|---|---|
| ~~10–12/11~~ | ✅ Repositório, dataset, EDA e split |
| **até 20/11** | Treinar os 6 experimentos (atrasado — prioridade) |
| 18–20/11 | Avaliação no teste, tabelas, gráficos, análise de erros |
| 21–23/11 | YOLO (extra) e Discussão |
| 24–25/11 | Revisão cruzada, ABNT, limite de 10 páginas |
| 26/11 | README final, modelos no Drive público, testar avaliação sem retreino |
| **27/11 23h59** | **Cada integrante envia o PDF no AVA** (sem PDF = nota zero) |

---

## Referência

YANG, Mindy; THUNG, Gary. **Classification of trash for recyclability status**. CS229 Project Report. Stanford: Stanford University, 2016.
