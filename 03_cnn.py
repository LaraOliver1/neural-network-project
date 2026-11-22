"""
03_cnn.py — Experimentos CNN (integrante 3; resoluções maiores: integrante 4)

Treina:
  cnn_128_sem   CNN, 128×96, sem pesos de classe            (padrão)
  cnn_128_com   CNN, 128×96, com pesos de classe            (padrão)
  cnn_256_sem   CNN, 256×192, sem pesos de classe           (integrante 4, via --exp)
  cnn_512_sem   CNN, 512×384, sem pesos de classe           (integrante 4, via --exp)

Usa o common.py (mesmo split, semente 42, mesmo pipeline e mesmos callbacks dos outros modelos).
O conjunto de TESTE não é usado aqui — só no 05_resultados.py, no final.

Uso (na raiz do projeto, com a pasta dataset-resized/ ao lado do split.json):
  python 03_cnn.py                                   # treina cnn_128_sem e cnn_128_com (pula o que já terminou)
  python 03_cnn.py --exp cnn_128_sem                 # só um
  python 03_cnn.py --forcar                          # treina de novo do zero
  python 03_cnn.py --exp cnn_256_sem cnn_512_sem     # resoluções maiores (integrante 4)
  python 03_cnn.py --exp cnn_512_sem --batch_size 16 # se faltar memória na GPU (registre e justifique)

Arquitetura (justificativa resumida, para o artigo):
  - Entrada: imagem altura × largura × 3 (96×128×3 em 128×96), pixels / 255.
  - 3 blocos [Conv2D 3×3 (32, 64, 128 filtros, padding "same") -> BatchNormalization -> ReLU
    -> MaxPooling 2×2], como sugere o enunciado (2 ou 3 blocos de convolução e pooling).
    As convoluções compartilham pesos entre posições e exploram a vizinhança entre pixels,
    o que o MLP não faz; o número de filtros dobra a cada bloco enquanto a imagem cai pela metade.
  - GlobalAveragePooling2D antes da classificação: em vez de achatar o mapa de características,
    tira a média de cada filtro. Assim o número de parâmetros é IDÊNTICO nas três resoluções
    (128, 256 e 512), o que isola o efeito da resolução (recomendação do enunciado), e a
    CNN fica com ~0,1 milhão de parâmetros contra ~9,5 milhões do MLP.
  - Dropout (0,3) antes e depois de uma Dense(128, ReLU) e regularização L2 (1e-4) nas
    convoluções contra overfitting (só 1.769 imagens de treino).
  - Saída Dense(6, softmax); perda sparse_categorical_crossentropy.
  - Adam com lr 1e-3 (padrão do Adam; com BatchNorm e poucos parâmetros o treino é estável),
    lote 32, ReduceLROnPlateau e EarlyStopping em val_loss (paciência 10), guardando os pesos
    da melhor época na validação (tudo vem do common.py, igual ao MLP).
  Os valores podem ser ajustados pelas opções abaixo, SEMPRE olhando só a validação
  (F1 macro de validação é a métrica de seleção pedida no enunciado). Se fizer busca de
  hiperparâmetros, registre cada configuração testada numa tabela separada.
  cnn_128_com usa exatamente a mesma configuração e semente do cnn_128_sem; só muda a perda,
  que passa a ser ponderada pelos pesos de classe (inversamente proporcionais à frequência no treino).
  cnn_256_sem e cnn_512_sem usam a mesma arquitetura e hiperparâmetros, com pesos novos; só
  mude o lote (--batch_size) se faltar memória, e justifique no artigo.
"""
import argparse

from common import DIR_HISTORICOS, carregar_historico, treinar_experimento

EXPERIMENTOS_CNN = ["cnn_128_sem", "cnn_128_com", "cnn_256_sem", "cnn_512_sem"]
PADRAO = ["cnn_128_sem", "cnn_128_com"]
EXPERIMENTOS_MLP = ["mlp_128_sem", "mlp_128_com"]


def construir_cnn(input_shape, num_classes, filtros=(32, 64, 128), dropout=0.3, l2=1e-4, densa=128):
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    entrada = tf.keras.Input(shape=input_shape)
    x = entrada
    for f in filtros:
        x = layers.Conv2D(f, 3, padding="same", kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(densa, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    saida = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(entrada, saida, name="cnn")


def _linha(nome, m):
    return (f"  {nome:12s} val_F1_macro={m['val_f1_macro']:.3f}  val_acc={m['val_accuracy']:.3f}  "
            f"época salva={m['epoca_do_modelo_salvo']}/{m['epocas_rodadas']}  "
            f"tempo={m['tempo_total_treino_s'] / 60:.1f} min ({m['tempo_medio_por_epoca_s']:.1f} s/época)  "
            f"params={m['parametros_treinaveis']:,}  {m['tamanho_modelo_salvo_mb']} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=EXPERIMENTOS_CNN, nargs="+", default=PADRAO)
    ap.add_argument("--split", default="split.json")
    ap.add_argument("--data_dir", default="dataset-resized")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--filtros", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--densa", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=None, help="padrão: o do common.py (32)")
    ap.add_argument("--epocas", type=int, default=None, help="padrão: o do common.py (100)")
    ap.add_argument("--forcar", action="store_true", help="treina de novo mesmo se já existir")
    args = ap.parse_args()

    def construir(input_shape, num_classes):
        return construir_cnn(input_shape, num_classes, args.filtros, args.dropout, args.l2, args.densa)

    for nome in args.exp:
        print("\n" + "=" * 70 + f"\n{nome}\n" + "=" * 70)
        treinar_experimento(
            nome, construir, lr=args.lr, split_path=args.split, data_dir=args.data_dir,
            batch_size=args.batch_size, epocas_max=args.epocas, forcar=args.forcar,
            info_extra={"arquitetura": {"filtros": args.filtros, "kernel": 3, "pooling": "max 2x2",
                                        "global_average_pooling": True, "densa": args.densa,
                                        "dropout": args.dropout, "l2": args.l2,
                                        "batchnorm": True, "ativacao": "relu"}})

    # Resumo de tudo que já foi treinado (inclusive o que foi pulado), com o MLP para comparação
    nomes = [n for n in EXPERIMENTOS_MLP + EXPERIMENTOS_CNN if (DIR_HISTORICOS / f"{n}.json").exists()]
    if nomes:
        print("\nResumo (VALIDAÇÃO — o teste fica para o 05_resultados.py):")
        for nome in nomes:
            print(_linha(nome, carregar_historico(nome)["meta"]))


if __name__ == "__main__":
    main()
