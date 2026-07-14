"""Treina uma U-Net para identificar pivôs centrais a partir de um arquivo ZIP.

Uso esperado no Colab ou localmente:

    python train_pivos_from_zip.py \
      --zip projects/sefazgogeoprocessamento/assets/pivo.zip \
      --epochs 40 \
      --output Modelo_UNet_Pivos.keras

O ZIP deve conter imagens e máscaras em pastas com nomes comuns, por exemplo:
`images/` e `masks/`, `imagem/` e `mascara/`, ou arquivos cujos caminhos
incluam `image|img|satellite|rgb|ndvi` para entradas e `mask|label|mascara|pivo`
para os gabaritos. Pares são associados pelo nome-base do arquivo.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_HINTS = ("image", "images", "img", "satellite", "rgb", "ndvi", "entrada", "imagem")
MASK_HINTS = ("mask", "masks", "label", "labels", "mascara", "máscara", "pivo", "pivots")


def normalize_stem(path: Path) -> str:
    """Normaliza nomes para parear imagem e máscara do mesmo pivô."""
    stem = path.stem.lower()
    for token in (*IMAGE_HINTS, *MASK_HINTS, "_", "-", " "):
        stem = stem.replace(token, "")
    return stem


def discover_pairs(extract_dir: Path) -> list[tuple[Path, Path]]:
    """Encontra pares de imagem/máscara dentro do ZIP extraído."""
    files = [p for p in extract_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    image_candidates: dict[str, Path] = {}
    mask_candidates: dict[str, Path] = {}

    for file_path in files:
        normalized_path = file_path.as_posix().lower()
        key = normalize_stem(file_path)
        if any(hint in normalized_path for hint in MASK_HINTS):
            mask_candidates.setdefault(key, file_path)
        elif any(hint in normalized_path for hint in IMAGE_HINTS):
            image_candidates.setdefault(key, file_path)

    pairs = sorted(
        (image_candidates[key], mask_candidates[key])
        for key in image_candidates.keys() & mask_candidates.keys()
    )
    if not pairs:
        raise ValueError(
            "Não encontrei pares de imagem e máscara no ZIP. Organize o arquivo com "
            "pastas como images/ e masks/ ou use nomes contendo image/img e mask/label."
        )
    return pairs


def load_image(path: Any, image_size: int, channels: int, tf: Any) -> Any:
    data = tf.io.read_file(path)
    image = tf.io.decode_image(data, channels=channels, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size], method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    return image


def load_mask(path: Any, image_size: int, tf: Any) -> Any:
    data = tf.io.read_file(path)
    mask = tf.io.decode_image(data, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, [image_size, image_size], method="nearest")
    mask = tf.cast(mask > 127, tf.float32)
    return mask


def make_dataset(
    pairs: list[tuple[Path, Path]], image_size: int, channels: int, batch_size: int
) -> tuple[Any, Any]:
    import tensorflow as tf

    image_paths = [str(image) for image, _ in pairs]
    mask_paths = [str(mask) for _, mask in pairs]
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.shuffle(len(pairs), seed=42, reshuffle_each_iteration=False)

    def parse(image_path: Any, mask_path: Any) -> tuple[Any, Any]:
        return load_image(image_path, image_size, channels, tf), load_mask(mask_path, image_size, tf)

    dataset = dataset.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    val_size = max(1, int(len(pairs) * 0.2)) if len(pairs) > 1 else 0
    train_ds = dataset.skip(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = dataset.take(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def build_unet(image_size: int, channels: int) -> Any:
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=(image_size, image_size, channels))

    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(c3)

    u4 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(64, 3, activation="relu", padding="same")(u4)
    c4 = layers.Conv2D(64, 3, activation="relu", padding="same")(c4)

    u5 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(32, 3, activation="relu", padding="same")(u5)
    c5 = layers.Conv2D(32, 3, activation="relu", padding="same")(c5)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c5)
    return models.Model(inputs, outputs)


def unzip_dataset(zip_path: Path, extract_dir: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Arquivo ZIP não encontrado: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"O arquivo informado não é um ZIP válido: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)


def train(args: argparse.Namespace) -> None:
    import tensorflow as tf
    from tensorflow.keras import callbacks

    temp_dir = Path(args.extract_dir) if args.extract_dir else Path(tempfile.mkdtemp(prefix="pivos_"))
    try:
        unzip_dataset(Path(args.zip), temp_dir)
        pairs = discover_pairs(temp_dir)
        print(f"✅ Pares imagem/máscara encontrados: {len(pairs)}")

        train_ds, val_ds = make_dataset(pairs, args.image_size, args.channels, args.batch_size)
        model = build_unet(args.image_size, args.channels)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        cb = [
            callbacks.ModelCheckpoint(args.output, save_best_only=True, monitor="val_loss"),
            callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True),
        ]
        validation_data = val_ds if len(pairs) > 1 else None
        model.fit(train_ds, validation_data=validation_data, epochs=args.epochs, callbacks=cb)
        model.save(args.output)
        print(f"✅ Modelo salvo em: {args.output}")
    finally:
        if not args.extract_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina U-Net para segmentação de pivôs a partir de ZIP.")
    parser.add_argument("--zip", default="projects/sefazgogeoprocessamento/assets/pivo.zip")
    parser.add_argument("--output", default="Modelo_UNet_Pivos.keras")
    parser.add_argument("--extract-dir", default=None, help="Diretório opcional para manter o ZIP extraído.")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    train(parse_args())
