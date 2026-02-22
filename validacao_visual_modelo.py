import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
from google.colab import drive

if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

print('--- INICIANDO PROVA REAL ---')
pasta_base = '/content/drive/MyDrive/Tese_IA_Jussara'
caminho_modelo = os.path.join(pasta_base, 'Modelo_UNet_Jussara_2025_FINAL_v2.keras')

if not os.path.exists(caminho_modelo):
    raise FileNotFoundError(f'Modelo não encontrado: {caminho_modelo}')

model = tf.keras.models.load_model(caminho_modelo, compile=False)
print(f'✅ Modelo carregado: {caminho_modelo}')

busca_dados = glob.glob(os.path.join(pasta_base, '*MASSIVE*tfrecord*'))
if not busca_dados:
    raise FileNotFoundError('Nenhum TFRecord MASSIVE encontrado para validação visual.')

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
        min_v = tf.reduce_min(img)
        max_v = tf.reduce_max(img)
        img = tf.where(max_v > min_v, (img - min_v) / (max_v - min_v + 1e-6), tf.zeros_like(img))
        inputs_list.append(img)

    image_stacked = tf.concat(inputs_list, axis=-1)
    dense_lbl = tf.sparse.to_dense(parsed[LABEL_BAND], default_value=0.0)
    lbl = tf.reshape(dense_lbl, [READ_SIZE, READ_SIZE, 1])
    lbl = tf.image.resize_with_crop_or_pad(lbl, KERNEL_SIZE, KERNEL_SIZE)
    lbl = tf.cast(lbl > 0.5, tf.float32)
    return image_stacked, lbl


dataset = tf.data.TFRecordDataset(caminho_dados, compression_type='GZIP').map(parse_fast).batch(10).take(1)
imgs, labels = next(iter(dataset))
preds = model.predict(imgs, verbose=0)
preds_bin = (preds > 0.5).astype('float32')

intersection = (preds_bin * labels.numpy()).sum(axis=(1, 2, 3))
union = ((preds_bin + labels.numpy()) > 0).sum(axis=(1, 2, 3))
iou = (intersection + 1e-6) / (union + 1e-6)
print(f'📏 IoU médio no lote: {iou.mean():.4f}')

plt.figure(figsize=(18, 14))
print('\nLEGENDA: Esquerda=Satélite | Meio=Gabarito | Direita=Predição binária')

n_show = min(5, imgs.shape[0])
for i in range(n_show):
    plt.subplot(n_show, 3, i * 3 + 1)
    plt.imshow(imgs[i][:, :, 2], cmap='RdYlGn', vmin=0, vmax=1)
    plt.axis('off')
    if i == 0:
        plt.title('Satélite (NDVI)')

    plt.subplot(n_show, 3, i * 3 + 2)
    plt.imshow(labels[i][:, :, 0], cmap='binary_r')
    plt.axis('off')
    if i == 0:
        plt.title('Gabarito Real')

    plt.subplot(n_show, 3, i * 3 + 3)
    plt.imshow(preds_bin[i][:, :, 0], cmap='viridis')
    plt.axis('off')
    if i == 0:
        plt.title('Predição (>0.5)')

saida_fig = os.path.join(pasta_base, 'prova_real_validacao.png')
plt.tight_layout()
plt.savefig(saida_fig, dpi=200, bbox_inches='tight')
plt.show()
print(f'🖼️ Prova real salva em: {saida_fig}')
