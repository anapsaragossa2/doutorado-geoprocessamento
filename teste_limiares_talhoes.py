import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import glob
import numpy as np
import tensorflow as tf


def garantir_drive_montado() -> str:
    pasta_base = '/content/drive/MyDrive/Tese_IA_Jussara'
    if os.path.exists('/content/drive'):
        return pasta_base
    raise RuntimeError(
        "Drive não montado. No Colab, execute antes:\n"
        "from google.colab import drive\n"
        "drive.mount('/content/drive')"
    )


def parse_thresholds(default):
    csv = os.environ.get('THRESHOLDS_CSV', '')
    if not csv.strip():
        return default
    vals = []
    for x in csv.split(','):
        x = x.strip()
        if x:
            vals.append(float(x))
    return vals if vals else default


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
print(f'📂 Dados usados no teste: {caminho_dados}')

KERNEL_SIZE = 128
READ_SIZE = 129
INPUT_BANDS = ['R_1', 'NIR_1', 'NDVI_1', 'R_2', 'NIR_2', 'NDVI_2']
LABEL_BAND = 'label_chip'
BATCH_SIZE = int(os.environ.get('BATCH_SIZE_SWEEP', '16'))
N_BATCHES = int(os.environ.get('N_BATCHES_SWEEP', '20'))
THRESHOLDS = parse_thresholds([0.30, 0.40, 0.50, 0.60, 0.70])
OUTPUT_FILENAME = os.environ.get('SWEEP_OUTPUT_FILENAME', 'resultado_teste_limiares.txt')


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


def metricas_binarias(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(np.uint8)
    y_true = y_true.astype(np.uint8)

    tp = np.logical_and(y_pred == 1, y_true == 1).sum()
    fp = np.logical_and(y_pred == 1, y_true == 0).sum()
    fn = np.logical_and(y_pred == 0, y_true == 1).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    return precision, recall, f1, iou


print('🤖 Carregando modelo...')
model = tf.keras.models.load_model(modelo_path, compile=False)

print('📦 Preparando lote de validação para sweep de limiar...')
dataset = tf.data.TFRecordDataset(caminho_dados, compression_type='GZIP').map(parse_fast).batch(BATCH_SIZE).take(N_BATCHES)

all_probs, all_labels = [], []
for imgs, labels in dataset:
    probs = model.predict(imgs, verbose=0)
    all_probs.append(probs)
    all_labels.append(labels.numpy())

if not all_probs:
    raise RuntimeError('Nenhum batch lido para teste de limiares.')

y_prob = np.concatenate(all_probs, axis=0)
y_true = np.concatenate(all_labels, axis=0)

print(f'✅ Amostras avaliadas: {y_prob.shape[0]}')
print('\n📊 Resultado por limiar:')
print('thr\tprecision\trecall\tf1\tiou')

best = None
for thr in THRESHOLDS:
    p, r, f1, iou = metricas_binarias(y_true, y_prob, thr)
    print(f'{thr:.2f}\t{p:.4f}\t\t{r:.4f}\t{f1:.4f}\t{iou:.4f}')
    if best is None or f1 > best['f1']:
        best = {'thr': thr, 'precision': p, 'recall': r, 'f1': f1, 'iou': iou}

print('\n🏆 Melhor limiar (por F1):')
print(best)

out_path = os.path.join(pasta_base, OUTPUT_FILENAME)
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('thr\tprecision\trecall\tf1\tiou\n')
    for thr in THRESHOLDS:
        p, r, f1, iou = metricas_binarias(y_true, y_prob, thr)
        f.write(f'{thr:.2f}\t{p:.6f}\t{r:.6f}\t{f1:.6f}\t{iou:.6f}\n')
    f.write(f"\nmelhor_thr={best['thr']:.2f}\n")

print(f'📝 Resultado salvo em: {out_path}')
