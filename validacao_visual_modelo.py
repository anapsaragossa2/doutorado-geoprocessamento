import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from google.colab import drive

# 1. Montar Drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

print("--- INICIANDO PROVA REAL ---")

# 2. Localizar o Modelo Salvo
pasta_base = '/content/drive/MyDrive/Tese_IA_Jussara'
caminho_modelo = os.path.join(pasta_base, 'Modelo_UNet_Jussara_2025_FINAL_v2.keras')

if os.path.exists(caminho_modelo):
    print(f"✅ Arquivo do modelo encontrado: {caminho_modelo}")

    # CARREGAR O MODELO (O momento da verdade)
    try:
        model = tf.keras.models.load_model(caminho_modelo)
        print("✅ Modelo carregado na memória com sucesso!")
    except Exception as e:
        print(f"❌ ERRO ao carregar modelo: {e}")
else:
    print("❌ ERRO: O arquivo .keras não foi encontrado no local esperado.")

# 3. Carregar um pouco de dados para testar
# (Precisamos das imagens para passar pelo modelo)
busca_dados = glob.glob(os.path.join(pasta_base, '*MASSIVE*tfrecord*'))
caminho_dados = busca_dados[0]

KERNEL_SIZE = 128
READ_SIZE = 129
INPUT_BANDS = ['R_1', 'NIR_1', 'NDVI_1', 'R_2', 'NIR_2', 'NDVI_2']
LABEL_BAND = 'label_chip'

def parse_fast(example_proto):
    features_dict = {band: tf.io.VarLenFeature(tf.float32) for band in INPUT_BANDS + [LABEL_BAND]}
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

# Pega apenas 1 lote de 10 imagens
dataset = tf.data.TFRecordDataset(caminho_dados, compression_type='GZIP')
dataset = dataset.map(parse_fast).batch(10).take(1)

# 4. Gerar Previsões
print("🔮 Gerando previsões com o modelo carregado...")
imgs, labels = next(iter(dataset))
preds = model.predict(imgs)

# 5. Visualizar
plt.figure(figsize=(15, 12))
print("\nLEGENDA: Esquerda=Satélite | Meio=Gabarito | Direita=O que a IA Aprendeu")

for i in range(5): # Mostra 5 exemplos
    # Satélite (NDVI Safra)
    plt.subplot(5, 3, i*3 + 1)
    plt.imshow(imgs[i][:,:,2], cmap='RdYlGn', vmin=0, vmax=0.8)
    plt.axis('off')
    if i==0: plt.title('Satélite (NDVI)')

    # Gabarito
    plt.subplot(5, 3, i*3 + 2)
    plt.imshow(labels[i][:,:,0], cmap='binary_r')
    plt.axis('off')
    if i==0: plt.title('Gabarito Real')

    # Predição da IA
    plt.subplot(5, 3, i*3 + 3)
    # Vmin/Vmax fixos para ver a confiança real
    plt.imshow(preds[i][:,:,0], cmap='magma', vmin=0, vmax=1)
    plt.axis('off')
    if i==0: plt.title('IA (Arquivo Salvo)')

plt.tight_layout()
plt.show()
