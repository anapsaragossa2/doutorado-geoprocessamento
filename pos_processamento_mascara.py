import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


def garantir_drive_montado() -> str:
    pasta_base = '/content/drive/MyDrive/Tese_IA_Jussara'
    if os.path.exists('/content/drive'):
        return pasta_base
    raise RuntimeError(
        "Drive não montado. No Colab, execute antes:\n"
        "from google.colab import drive\n"
        "drive.mount('/content/drive')"
    )


def limpar_mascara(mask_bin: np.ndarray, min_size: int = 100) -> np.ndarray:
    mask = mask_bin.astype(bool)
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))

    labeled, n = ndi.label(mask)
    if n == 0:
        return mask.astype(np.uint8)

    counts = np.bincount(labeled.ravel())
    remove = counts < min_size
    remove[0] = False
    mask[remove[labeled]] = False
    return mask.astype(np.uint8)


def metricas(y_true: np.ndarray, y_pred: np.ndarray):
    tp = np.logical_and(y_pred == 1, y_true == 1).sum()
    fp = np.logical_and(y_pred == 1, y_true == 0).sum()
    fn = np.logical_and(y_pred == 0, y_true == 1).sum()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    return precision, recall, f1, iou


pasta_base = garantir_drive_montado()
modelo_path = os.path.join(pasta_base, 'Modelo_UNet_Jussara_2025_FINAL_v2.keras')
if not os.path.exists(modelo_path):
    raise FileNotFoundError(f'Modelo não encontrado: {modelo_path}')

busca = glob.glob(os.path.join(pasta_base, '*MASSIVE*tfrecord*'))
if not busca:
    busca = glob.glob(os.path.join(pasta_base, '*tfrecord*'))
if not busca:
    raise FileNotFoundError(f'Nenhum TFRecord encontrado em: {pasta_base}')

caminho_dados = sorted(busca, key=os.path.getmtime, reverse=True)[0]
print(f'📂 Dados usados: {caminho_dados}')

KERNEL_SIZE = 128
READ_SIZE = 129
INPUT_BANDS = ['R_1', 'NIR_1', 'NDVI_1', 'R_2', 'NIR_2', 'NDVI_2']
LABEL_BAND = 'label_chip'
THRESHOLD = float(os.environ.get('THRESHOLD_INFERENCIA', '0.30'))
MIN_SIZE = int(os.environ.get('MIN_COMPONENT_SIZE', '100'))


def parse_fast(example_proto):
    features = {band: tf.io.VarLenFeature(tf.float32) for band in INPUT_BANDS + [LABEL_BAND]}
    parsed = tf.io.parse_single_example(example_proto, features)
    xs = []
    for band in INPUT_BANDS:
        dense = tf.sparse.to_dense(parsed[band], default_value=0.0)
        img = tf.reshape(dense, [READ_SIZE, READ_SIZE, 1])
        img = tf.image.resize_with_crop_or_pad(img, KERNEL_SIZE, KERNEL_SIZE)
        min_v = tf.reduce_min(img)
        max_v = tf.reduce_max(img)
        img = tf.where(max_v > min_v, (img - min_v) / (max_v - min_v + 1e-6), tf.zeros_like(img))
        xs.append(img)
    image = tf.concat(xs, axis=-1)

    lbl = tf.sparse.to_dense(parsed[LABEL_BAND], default_value=0.0)
    lbl = tf.reshape(lbl, [READ_SIZE, READ_SIZE, 1])
    lbl = tf.image.resize_with_crop_or_pad(lbl, KERNEL_SIZE, KERNEL_SIZE)
    lbl = tf.cast(lbl > 0.5, tf.float32)
    return image, lbl


model = tf.keras.models.load_model(modelo_path, compile=False)
dataset = tf.data.TFRecordDataset(caminho_dados, compression_type='GZIP').map(parse_fast).batch(10).take(1)
imgs, labels = next(iter(dataset))
probs = model.predict(imgs, verbose=0)
raw = (probs > THRESHOLD).astype(np.uint8)

clean = np.zeros_like(raw)
for i in range(raw.shape[0]):
    clean[i, :, :, 0] = limpar_mascara(raw[i, :, :, 0], min_size=MIN_SIZE)

y_true = labels.numpy().astype(np.uint8)

p_raw, r_raw, f_raw, i_raw = metricas(y_true, raw)
p_cln, r_cln, f_cln, i_cln = metricas(y_true, clean)

print('📊 Métricas antes do pós-processamento:')
print(f'precision={p_raw:.4f} recall={r_raw:.4f} f1={f_raw:.4f} iou={i_raw:.4f}')
print('📊 Métricas depois do pós-processamento:')
print(f'precision={p_cln:.4f} recall={r_cln:.4f} f1={f_cln:.4f} iou={i_cln:.4f}')

plt.figure(figsize=(18, 16))
n_show = min(4, imgs.shape[0])
for i in range(n_show):
    plt.subplot(n_show, 4, i*4 + 1)
    plt.imshow(imgs[i][:,:,2], cmap='RdYlGn', vmin=0, vmax=1)
    plt.axis('off')
    if i == 0: plt.title('Satélite NDVI')

    plt.subplot(n_show, 4, i*4 + 2)
    plt.imshow(y_true[i,:,:,0], cmap='binary_r')
    plt.axis('off')
    if i == 0: plt.title('Gabarito')

    plt.subplot(n_show, 4, i*4 + 3)
    plt.imshow(raw[i,:,:,0], cmap='magma')
    plt.axis('off')
    if i == 0: plt.title(f'Bruta > {THRESHOLD:.2f}')

    plt.subplot(n_show, 4, i*4 + 4)
    plt.imshow(clean[i,:,:,0], cmap='viridis')
    plt.axis('off')
    if i == 0: plt.title(f'Pós-processada (min_size={MIN_SIZE})')

plt.tight_layout()
out_img = os.path.join(pasta_base, 'comparativo_pos_processamento.png')
plt.savefig(out_img, dpi=200, bbox_inches='tight')
plt.show()
print(f'🖼️ Comparativo salvo em: {out_img}')

out_metrics = os.path.join(pasta_base, 'resultado_pos_processamento.txt')
with open(out_metrics, 'w', encoding='utf-8') as f:
    f.write('stage\tprecision\trecall\tf1\tiou\n')
    f.write(f'antes\t{p_raw:.6f}\t{r_raw:.6f}\t{f_raw:.6f}\t{i_raw:.6f}\n')
    f.write(f'depois\t{p_cln:.6f}\t{r_cln:.6f}\t{f_cln:.6f}\t{i_cln:.6f}\n')
    f.write(f'threshold={THRESHOLD:.2f}\n')
    f.write(f'min_component_size={MIN_SIZE}\n')
print(f'📝 Métricas do pós-processamento salvas em: {out_metrics}')
