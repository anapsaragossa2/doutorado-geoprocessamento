import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks


def garantir_drive_montado() -> str:
    pasta_base = '/content/drive/MyDrive/Tese_IA_Jussara'
    if os.path.exists('/content/drive'):
        return pasta_base

    raise RuntimeError(
        "Drive não montado. No Colab, execute antes: \n"
        "from google.colab import drive\n"
        "drive.mount('/content/drive')"
    )


pasta_base = garantir_drive_montado()

KERNEL_SIZE = 128
READ_SIZE = 129
BATCH_SIZE = 32
EPOCHS = 120
SEED = 42
INPUT_BANDS = ['R_1', 'NIR_1', 'NDVI_1', 'R_2', 'NIR_2', 'NDVI_2']
LABEL_BAND = 'label_chip'


def localizar_tfrecord(base_dir: str) -> str:
    busca = glob.glob(os.path.join(base_dir, '*MASSIVE*tfrecord*'))
    if not busca:
        busca = glob.glob(os.path.join(base_dir, '*tfrecord*'))
        busca.sort(key=os.path.getmtime, reverse=True)
    if not busca:
        raise FileNotFoundError(f'Nenhum TFRecord encontrado em {base_dir}')
    return busca[0]


def normalizar_banda(img: tf.Tensor) -> tf.Tensor:
    min_v = tf.reduce_min(img)
    max_v = tf.reduce_max(img)
    return tf.where(max_v > min_v, (img - min_v) / (max_v - min_v + 1e-6), tf.zeros_like(img))


def parse_and_process(example_proto):
    features_dict = {
        band: tf.io.VarLenFeature(tf.float32) for band in INPUT_BANDS + [LABEL_BAND]
    }
    parsed = tf.io.parse_single_example(example_proto, features_dict)

    inputs_list = []
    for band in INPUT_BANDS:
        dense = tf.sparse.to_dense(parsed[band], default_value=0.0)
        img = tf.reshape(dense, [READ_SIZE, READ_SIZE, 1])
        img = tf.image.resize_with_crop_or_pad(img, KERNEL_SIZE, KERNEL_SIZE)
        img = normalizar_banda(img)
        inputs_list.append(img)

    image_stacked = tf.concat(inputs_list, axis=-1)

    dense_lbl = tf.sparse.to_dense(parsed[LABEL_BAND], default_value=0.0)
    lbl = tf.reshape(dense_lbl, [READ_SIZE, READ_SIZE, 1])
    lbl = tf.image.resize_with_crop_or_pad(lbl, KERNEL_SIZE, KERNEL_SIZE)
    lbl = tf.cast(lbl > 0.5, tf.float32)
    return image_stacked, lbl


def augment(image, label):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        label = tf.image.flip_up_down(label)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    label = tf.image.rot90(label, k)
    return image, label


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice_loss = 1.0 - dice_coef(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def build_unet(input_shape):
    inputs = layers.Input(shape=input_shape)

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 256)

    u5 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = conv_block(u5, 128)

    u6 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = conv_block(u6, 64)

    u7 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = conv_block(u7, 32)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c7)
    return models.Model(inputs=[inputs], outputs=[outputs])


caminho_arquivo = localizar_tfrecord(pasta_base)
print(f"📂 Lendo dados de: {caminho_arquivo}")

print("🔢 Verificando tamanho do arquivo...")
raw_dataset = tf.data.TFRecordDataset(caminho_arquivo, compression_type='GZIP')
N_REAL = sum(1 for _ in raw_dataset)
print(f"✅ Total de amostras: {N_REAL}")

N_TRAIN = int(N_REAL * 0.8)
full_dataset = tf.data.TFRecordDataset(caminho_arquivo, compression_type='GZIP').map(
    parse_and_process, num_parallel_calls=tf.data.AUTOTUNE
)
full_dataset = full_dataset.shuffle(max(N_REAL, 1), seed=SEED, reshuffle_each_iteration=False)

train_ds = full_dataset.take(N_TRAIN).map(augment, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(max(N_TRAIN, 1), seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = full_dataset.skip(N_TRAIN).cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

model = build_unet((KERNEL_SIZE, KERNEL_SIZE, len(INPUT_BANDS)))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=bce_dice_loss,
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name='iou'),
        dice_coef,
    ],
)



def calcular_matriz_confusao(model, dataset, threshold: float, max_batches: int = 20):
    tp = fp = fn = tn = 0
    for batch_idx, (x_batch, y_batch) in enumerate(dataset):
        if batch_idx >= max_batches:
            break
        probs = model.predict(x_batch, verbose=0)
        y_pred = (probs > threshold).astype(np.uint8)
        y_true = y_batch.numpy().astype(np.uint8)

        tp += np.logical_and(y_pred == 1, y_true == 1).sum()
        fp += np.logical_and(y_pred == 1, y_true == 0).sum()
        fn += np.logical_and(y_pred == 0, y_true == 1).sum()
        tn += np.logical_and(y_pred == 0, y_true == 0).sum()

    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def salvar_matriz_confusao(cm: np.ndarray, out_png: str, out_txt: str):
    total = cm.sum() + 1e-9
    cm_norm = cm / total

    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write('matriz_confusao_pixel\n')
        f.write('\tpred_0\tpred_1\n')
        f.write(f'true_0\t{cm[0,0]}\t{cm[0,1]}\n')
        f.write(f'true_1\t{cm[1,0]}\t{cm[1,1]}\n')
        f.write(f'total_pixels={int(total-1e-9)}\n')

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred 0', 'Pred 1'])
    ax.set_yticklabels(['True 0', 'True 1'])
    ax.set_title('Matriz de confusão (validação)')

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]*100:.2f}%)',
                    ha='center', va='center', color='black', fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)

checkpoint_last = os.path.join(pasta_base, 'Modelo_Checkpoint_last.keras')
checkpoint_best = os.path.join(pasta_base, 'Modelo_Checkpoint_best.keras')
csv_log = os.path.join(pasta_base, 'historico_treinamento.csv')
final_path = os.path.join(pasta_base, 'Modelo_UNet_Jussara_2025_FINAL_v2.keras')

cbs = [
    callbacks.ModelCheckpoint(checkpoint_last, save_best_only=False, verbose=1),
    callbacks.ModelCheckpoint(checkpoint_best, monitor='val_iou', mode='max', save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor='val_iou', mode='max', patience=12, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    callbacks.CSVLogger(csv_log),
]

print("🔥 Iniciando Retreinamento Rigoroso (foco em talhões)...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=cbs,
)

model.save(final_path)
print(f"✅ SUCESSO! Modelo final salvo em: {final_path}")
print(f"✅ Melhor checkpoint salvo em: {checkpoint_best}")

THRESHOLD_CM = float(os.environ.get('THRESHOLD_INFERENCIA', '0.30'))
MAX_BATCHES_CM = int(os.environ.get('CM_MAX_BATCHES', '20'))
cm = calcular_matriz_confusao(model, val_ds, threshold=THRESHOLD_CM, max_batches=MAX_BATCHES_CM)
cm_png = os.path.join(pasta_base, 'matriz_confusao_validacao.png')
cm_txt = os.path.join(pasta_base, 'matriz_confusao_validacao.txt')
salvar_matriz_confusao(cm, cm_png, cm_txt)
print(f'🧪 Matriz de confusão salva em: {cm_png}')
print(f'🧪 Valores da matriz de confusão salvos em: {cm_txt}')
