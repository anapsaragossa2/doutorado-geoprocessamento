import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import os
import glob
from google.colab import drive

# 1. Montar Drive (Obrigatório)
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# --- CONFIGURAÇÕES ---
pasta_base = '/content/drive/MyDrive/Tese_IA_Jussara'

# Busca o arquivo de dados JÁ EXISTENTE
busca = glob.glob(os.path.join(pasta_base, '*MASSIVE*tfrecord*'))
if not busca:
    # Se não achar o MASSIVE, pega qualquer um recente
    busca = glob.glob(os.path.join(pasta_base, '*tfrecord*'))
    busca.sort(key=os.path.getmtime, reverse=True)

caminho_arquivo = busca[0]
print(f"📂 Lendo dados de: {caminho_arquivo}")

# Parâmetros
KERNEL_SIZE = 128
READ_SIZE = 129
BATCH_SIZE = 32
EPOCHS = 40
INPUT_BANDS = ['R_1', 'NIR_1', 'NDVI_1', 'R_2', 'NIR_2', 'NDVI_2']
LABEL_BAND = 'label_chip'

# --- PIPELINE DE DADOS (Rápido) ---
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
        inputs_list.append(img)

    image_stacked = tf.concat(inputs_list, axis=-1)

    dense_lbl = tf.sparse.to_dense(parsed[LABEL_BAND], default_value=0.0)
    lbl = tf.reshape(dense_lbl, [READ_SIZE, READ_SIZE, 1])
    lbl = tf.image.resize_with_crop_or_pad(lbl, KERNEL_SIZE, KERNEL_SIZE)
    return image_stacked, lbl

# Contagem rápida para não dar erro de tamanho
print("🔢 Verificando tamanho do arquivo...")
raw_dataset = tf.data.TFRecordDataset(caminho_arquivo, compression_type='GZIP')
N_REAL = sum(1 for _ in raw_dataset)
print(f"✅ Total de amostras: {N_REAL}")

N_TRAIN = int(N_REAL * 0.8)
full_dataset = tf.data.TFRecordDataset(caminho_arquivo, compression_type='GZIP').map(parse_and_process)

train_ds = full_dataset.take(N_TRAIN).cache().shuffle(N_TRAIN).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = full_dataset.skip(N_TRAIN).cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- MODELO U-NET ---
def build_unet(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Camadas (Encoder)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs); p1 = layers.MaxPooling2D()(c1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1); p2 = layers.MaxPooling2D()(c2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2); p3 = layers.MaxPooling2D()(c3)

    # Bottleneck
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)

    # Decoder
    u5 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u5)

    u6 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)

    u7 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u7)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)
    return models.Model(inputs=[inputs], outputs=[outputs])

model = build_unet((KERNEL_SIZE, KERNEL_SIZE, 6))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- 🛡️ SALVAMENTO AUTOMÁTICO (SEGURANÇA) ---
# Salva um backup no Drive a cada melhoria
checkpoint_path = os.path.join(pasta_base, 'Modelo_Checkpoint.keras')
checkpoint_cb = callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=False, # Salva sempre o último estado
    verbose=1
)

print("🔥 Iniciando Retreinamento (Vai salvar automaticamente no Drive)...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb] # <--- Aqui está a segurança
)

# Salvamento Final Definitivo
final_path = os.path.join(pasta_base, 'Modelo_UNet_Jussara_2025_FINAL_v2.keras')
model.save(final_path)
print(f"✅ SUCESSO! Modelo final salvo em: {final_path}")
