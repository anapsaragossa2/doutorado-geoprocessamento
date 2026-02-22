"""Validação visual do modelo salvo para segmentação de pivôs."""

import glob
import os

import matplotlib.pyplot as plt
import tensorflow as tf

PASTA_BASE = "/content/drive/MyDrive/Tese_IA_Jussara"
CAMINHO_MODELO = os.path.join(PASTA_BASE, "Modelo_UNet_Jussara_2025_FINAL_v2.keras")
KERNEL_SIZE = 128
READ_SIZE = 129
INPUT_BANDS = ["R_1", "NIR_1", "NDVI_1", "R_2", "NIR_2", "NDVI_2"]
LABEL_BAND = "label_chip"


def montar_drive_se_necessario() -> None:
    if os.path.exists("/content/drive"):
        return
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")


def parse_fast(example_proto: tf.Tensor):
    features_dict = {
        band: tf.io.VarLenFeature(tf.float32) for band in INPUT_BANDS + [LABEL_BAND]
    }
    parsed = tf.io.parse_single_example(example_proto, features_dict)

    inputs_list = []
    for band in INPUT_BANDS:
        dense = tf.sparse.to_dense(parsed[band], default_value=0.0)
        img = tf.reshape(dense, [READ_SIZE, READ_SIZE, 1])
        img = tf.image.resize_with_crop_or_pad(img, KERNEL_SIZE, KERNEL_SIZE)
        inputs_list.append(img)

    image_stacked = tf.concat(inputs_list, axis=-1)

    dense_lbl = tf.sparse.to_dense(parsed[LABEL_BAND], default_value=0.0)
    lbl = tf.reshape(dense_lbl, [READ_SIZE, READ_SIZE, 1])
    lbl = tf.image.resize_with_crop_or_pad(lbl, KERNEL_SIZE, KERNEL_SIZE)
    return image_stacked, lbl


def main() -> None:
    montar_drive_se_necessario()

    print("--- INICIANDO PROVA REAL ---")
    if not os.path.exists(CAMINHO_MODELO):
        raise FileNotFoundError(f"Modelo não encontrado: {CAMINHO_MODELO}")

    print(f"✅ Arquivo do modelo encontrado: {CAMINHO_MODELO}")
    model = tf.keras.models.load_model(CAMINHO_MODELO)
    print("✅ Modelo carregado na memória com sucesso!")

    busca_dados = glob.glob(os.path.join(PASTA_BASE, "*MASSIVE*tfrecord*"))
    if not busca_dados:
        raise FileNotFoundError("Nenhum TFRecord MASSIVE encontrado para validação visual.")
    caminho_dados = sorted(busca_dados, key=os.path.getmtime, reverse=True)[0]

    dataset = tf.data.TFRecordDataset(caminho_dados, compression_type="GZIP")
    dataset = dataset.map(parse_fast, num_parallel_calls=tf.data.AUTOTUNE).batch(10).take(1)

    print("🔮 Gerando previsões com o modelo carregado...")
    imgs, labels = next(iter(dataset))
    preds = model.predict(imgs, verbose=0)

    plt.figure(figsize=(15, 12))
    print("\nLEGENDA: Esquerda=Satélite | Meio=Gabarito | Direita=O que a IA Aprendeu")

    n_show = min(5, imgs.shape[0])
    for i in range(n_show):
        plt.subplot(n_show, 3, i * 3 + 1)
        plt.imshow(imgs[i][:, :, 2], cmap="RdYlGn", vmin=0, vmax=0.8)
        plt.axis("off")
        if i == 0:
            plt.title("Satélite (NDVI)")

        plt.subplot(n_show, 3, i * 3 + 2)
        plt.imshow(labels[i][:, :, 0], cmap="binary_r")
        plt.axis("off")
        if i == 0:
            plt.title("Gabarito Real")

        plt.subplot(n_show, 3, i * 3 + 3)
        plt.imshow(preds[i][:, :, 0], cmap="magma", vmin=0, vmax=1)
        plt.axis("off")
        if i == 0:
            plt.title("IA (Arquivo Salvo)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
