"""Retreinamento seguro de U-Net para segmentação de pivôs.

Uso:
    python treinamento_seguro_unet.py

O script:
- localiza automaticamente TFRecord mais adequado;
- cria pipeline otimizado com tf.data;
- salva checkpoints (último e melhor modelo);
- permite retomar treino de um checkpoint existente.
"""

import glob
import os
from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models


@dataclass
class Config:
    pasta_base: str = os.environ.get("TESE_IA_BASE_DIR", "/workspace/doutorado-geoprocessamento")
    kernel_size: int = 128
    read_size: int = 129
    batch_size: int = 32
    epochs: int = 40
    validation_split: float = 0.2
    input_bands: tuple[str, ...] = (
        "R_1",
        "NIR_1",
        "NDVI_1",
        "R_2",
        "NIR_2",
        "NDVI_2",
    )
    label_band: str = "label_chip"


def localizar_tfrecord(pasta_base: str) -> str:
    busca_massive = glob.glob(os.path.join(pasta_base, "*MASSIVE*tfrecord*"))
    if busca_massive:
        return sorted(busca_massive, key=os.path.getmtime, reverse=True)[0]

    busca_geral = glob.glob(os.path.join(pasta_base, "*tfrecord*"))
    if not busca_geral:
        raise FileNotFoundError(f"Nenhum arquivo .tfrecord encontrado em: {pasta_base}")
    return sorted(busca_geral, key=os.path.getmtime, reverse=True)[0]


def parse_and_process(example_proto: tf.Tensor, cfg: Config):
    features_dict = {
        band: tf.io.VarLenFeature(tf.float32)
        for band in list(cfg.input_bands) + [cfg.label_band]
    }
    parsed = tf.io.parse_single_example(example_proto, features_dict)

    inputs_list = []
    for band in cfg.input_bands:
        dense = tf.sparse.to_dense(parsed[band], default_value=0.0)
        img = tf.reshape(dense, [cfg.read_size, cfg.read_size, 1])
        img = tf.image.resize_with_crop_or_pad(img, cfg.kernel_size, cfg.kernel_size)
        inputs_list.append(img)

    image_stacked = tf.concat(inputs_list, axis=-1)

    dense_lbl = tf.sparse.to_dense(parsed[cfg.label_band], default_value=0.0)
    lbl = tf.reshape(dense_lbl, [cfg.read_size, cfg.read_size, 1])
    lbl = tf.image.resize_with_crop_or_pad(lbl, cfg.kernel_size, cfg.kernel_size)
    return image_stacked, lbl


def build_unet(input_shape: tuple[int, int, int]) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)

    c1 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    p1 = layers.MaxPooling2D()(c1)
    c2 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(p1)
    p2 = layers.MaxPooling2D()(c2)
    c3 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(p2)
    p3 = layers.MaxPooling2D()(c3)

    c4 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(p3)

    u5 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(u5)

    u6 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(u6)

    u7 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(u7)

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(c7)
    model = models.Model(inputs=[inputs], outputs=[outputs], name="unet_pivos")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    cfg = Config()

    print(f"📁 Pasta base configurada: {cfg.pasta_base}")
    os.makedirs(cfg.pasta_base, exist_ok=True)

    caminho_arquivo = localizar_tfrecord(cfg.pasta_base)
    print(f"📂 Lendo dados de: {caminho_arquivo}")

    print("🔢 Verificando tamanho do arquivo...")
    raw_dataset = tf.data.TFRecordDataset(caminho_arquivo, compression_type="GZIP")
    n_real = sum(1 for _ in raw_dataset)
    if n_real < 2:
        raise ValueError("Poucas amostras para treinar/validar (mínimo: 2).")
    print(f"✅ Total de amostras: {n_real}")

    n_train = int(n_real * (1 - cfg.validation_split))
    n_train = max(1, min(n_train, n_real - 1))

    full_dataset = tf.data.TFRecordDataset(caminho_arquivo, compression_type="GZIP")
    full_dataset = full_dataset.map(
        lambda x: parse_and_process(x, cfg), num_parallel_calls=tf.data.AUTOTUNE
    )

    train_ds = (
        full_dataset.take(n_train)
        .cache()
        .shuffle(min(n_train, 4096), reshuffle_each_iteration=True)
        .batch(cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        full_dataset.skip(n_train)
        .cache()
        .batch(cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = build_unet((cfg.kernel_size, cfg.kernel_size, len(cfg.input_bands)))
    model.summary()

    checkpoint_last = os.path.join(cfg.pasta_base, "Modelo_Checkpoint_last.keras")
    checkpoint_best = os.path.join(cfg.pasta_base, "Modelo_Checkpoint_best.keras")
    csv_log = os.path.join(cfg.pasta_base, "historico_treinamento.csv")
    final_path = os.path.join(cfg.pasta_base, "Modelo_UNet_Jussara_2025_FINAL_v2.keras")

    print("🧭 Arquivos de saída:")
    print(f"   - Checkpoint (último): {checkpoint_last}")
    print(f"   - Checkpoint (melhor): {checkpoint_best}")
    print(f"   - Log CSV: {csv_log}")
    print(f"   - Modelo final: {final_path}")

    if os.path.exists(checkpoint_last):
        print(f"♻️ Retomando treino de checkpoint: {checkpoint_last}")
        model = tf.keras.models.load_model(checkpoint_last)

    cbs = [
        callbacks.ModelCheckpoint(filepath=checkpoint_last, save_best_only=False, verbose=1),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_best,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
        ),
        callbacks.CSVLogger(csv_log, append=True),
    ]

    print("🔥 Iniciando Retreinamento Seguro...")
    model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, callbacks=cbs, verbose=1)

    model.save(final_path)
    print(f"✅ SUCESSO! Modelo final salvo em: {final_path}")


if __name__ == "__main__":
    main()
