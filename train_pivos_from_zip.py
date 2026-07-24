"""Treina uma U-Net simples para segmentação de pivôs a partir de um ZIP.

Estrutura esperada dentro do ZIP (nomes configuráveis por CLI):

    images/xxx.png   ou imagens/xxx.jpg
    masks/xxx.png    ou mascaras/xxx.jpg

Os arquivos de imagem e máscara são pareados pelo mesmo nome-base (`xxx`).
Este script foi pensado para Colab, mas também roda localmente quando TensorFlow
está instalado.
"""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def extract_zip(zip_path: Path, output_dir: Path, overwrite: bool = False) -> Path:
    """Extrai o ZIP de treinamento para `output_dir`."""
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP de treinamento não encontrado: {zip_path}")

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)

    return output_dir


def find_data_dir(root: Path, images_dir: str, masks_dir: str) -> Path:
    """Localiza a pasta que contém os subdiretórios de imagens e máscaras."""
    candidates = [root] + [path for path in root.rglob("*") if path.is_dir()]
    for candidate in candidates:
        if (candidate / images_dir).is_dir() and (candidate / masks_dir).is_dir():
            return candidate
    raise FileNotFoundError(
        "Não encontrei a estrutura esperada no ZIP. "
        f"Procurei subpastas '{images_dir}/' e '{masks_dir}/' em {root}."
    )


def list_pairs(data_dir: Path, images_dir: str, masks_dir: str) -> list[tuple[Path, Path]]:
    """Retorna pares imagem/máscara com o mesmo nome-base."""
    image_root = data_dir / images_dir
    mask_root = data_dir / masks_dir

    images = {
        path.stem: path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    masks = {
        path.stem: path
        for path in mask_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }

    common = sorted(images.keys() & masks.keys())
    if not common:
        raise FileNotFoundError(
            "Nenhum par imagem/máscara foi encontrado. "
            "Confirme se os arquivos têm o mesmo nome-base nas duas pastas."
        )

    return [(images[name], masks[name]) for name in common]


def decode_image(path: tf.Tensor, channels: int, image_size: tuple[int, int]) -> tf.Tensor:
    """Lê PNG/JPEG e normaliza para [0, 1]."""
    content = tf.io.read_file(path)
    image = tf.io.decode_image(content, channels=channels, expand_animations=False)
    image = tf.image.resize(image, image_size, method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    return image


def build_dataset(
    pairs: Sequence[tuple[Path, Path]],
    image_size: tuple[int, int],
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    """Cria um `tf.data.Dataset` de pares imagem/máscara."""
    image_paths = [str(image) for image, _ in pairs]
    mask_paths = [str(mask) for _, mask in pairs]
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    if shuffle:
        dataset = dataset.shuffle(len(pairs), seed=42, reshuffle_each_iteration=True)

    def load_pair(image_path: tf.Tensor, mask_path: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = decode_image(image_path, channels=3, image_size=image_size)
        mask = decode_image(mask_path, channels=1, image_size=image_size)
        mask = tf.cast(mask > 0.5, tf.float32)
        return image, mask

    return dataset.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def conv_block(inputs: tf.Tensor, filters: int) -> tf.Tensor:
    """Bloco convolucional padrão da U-Net."""
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x


def build_unet(image_size: tuple[int, int]) -> tf.keras.Model:
    """Monta uma U-Net compacta para segmentação binária."""
    inputs = layers.Input(shape=(image_size[0], image_size[1], 3))

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)

    bridge = conv_block(p3, 256)

    u3 = layers.UpSampling2D()(bridge)
    u3 = layers.Concatenate()([u3, c3])
    c4 = conv_block(u3, 128)
    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = conv_block(u2, 64)
    u1 = layers.UpSampling2D()(c5)
    u1 = layers.Concatenate()([u1, c1])
    c6 = conv_block(u1, 32)

    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="pivo_mask")(c6)
    model = models.Model(inputs, outputs, name="unet_pivos")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)],
    )
    return model


def split_pairs(pairs: Sequence[tuple[Path, Path]], validation_fraction: float) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Divide pares em treino e validação sem aleatoriedade implícita."""
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction deve estar entre 0 e 0.5.")

    validation_count = max(1, int(len(pairs) * validation_fraction))
    train_count = len(pairs) - validation_count
    if train_count < 1:
        raise ValueError("São necessários pelo menos dois pares imagem/máscara para treinar.")

    return list(pairs[:train_count]), list(pairs[train_count:])


def parse_image_size(value: str) -> tuple[int, int]:
    """Converte `128` ou `128x128` em tupla de tamanho."""
    if "x" in value.lower():
        height, width = value.lower().split("x", maxsplit=1)
        return int(height), int(width)
    size = int(value)
    return size, size


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Treina U-Net de pivôs a partir de um arquivo ZIP.")
    parser.add_argument("--zip", required=True, type=Path, help="Caminho do ZIP com imagens e máscaras.")
    parser.add_argument("--workdir", default=Path("/content/pivos_dataset"), type=Path, help="Pasta de extração do ZIP.")
    parser.add_argument("--images-dir", default="images", help="Nome da pasta de imagens dentro do ZIP.")
    parser.add_argument("--masks-dir", default="masks", help="Nome da pasta de máscaras dentro do ZIP.")
    parser.add_argument("--image-size", default="128", help="Tamanho de entrada, por exemplo 128 ou 256x256.")
    parser.add_argument("--batch-size", default=16, type=int, help="Tamanho do batch.")
    parser.add_argument("--epochs", default=30, type=int, help="Número máximo de épocas.")
    parser.add_argument("--validation-fraction", default=0.2, type=float, help="Fração dos pares usada para validação.")
    parser.add_argument("--output", default=Path("Modelo_UNet_Pivos.keras"), type=Path, help="Arquivo .keras de saída.")
    parser.add_argument("--overwrite", action="store_true", help="Reextrai o ZIP apagando o workdir existente.")
    return parser


def main(argv: Iterable[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    image_size = parse_image_size(args.image_size)

    extracted_dir = extract_zip(args.zip, args.workdir, overwrite=args.overwrite)
    data_dir = find_data_dir(extracted_dir, args.images_dir, args.masks_dir)
    pairs = list_pairs(data_dir, args.images_dir, args.masks_dir)
    train_pairs, validation_pairs = split_pairs(pairs, args.validation_fraction)

    print(f"Pares encontrados: {len(pairs)}")
    print(f"Treino: {len(train_pairs)} | Validação: {len(validation_pairs)}")
    print(f"Diretório de dados: {data_dir}")

    train_ds = build_dataset(train_pairs, image_size, args.batch_size, shuffle=True)
    validation_ds = build_dataset(validation_pairs, image_size, args.batch_size, shuffle=False)

    model = build_unet(image_size)
    model.summary()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    training_callbacks = [
        callbacks.ModelCheckpoint(args.output, monitor="val_loss", save_best_only=True),
        callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5),
    ]

    model.fit(train_ds, validation_data=validation_ds, epochs=args.epochs, callbacks=training_callbacks)
    model.save(args.output)
    print(f"Modelo salvo em: {args.output}")
    return args.output


if __name__ == "__main__":
    main()
