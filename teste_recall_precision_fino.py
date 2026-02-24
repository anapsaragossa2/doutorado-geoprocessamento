import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import glob
import math
import numpy as np
import tensorflow as tf
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


def localizar_tfrecord(base_dir: str) -> str:
    busca = glob.glob(os.path.join(base_dir, '*MASSIVE*tfrecord*'))
    if not busca:
        busca = glob.glob(os.path.join(base_dir, '*tfrecord*'))
    if not busca:
        raise FileNotFoundError(f'Nenhum TFRecord encontrado em: {base_dir}')
    return sorted(busca, key=os.path.getmtime, reverse=True)[0]


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
    lbl = tf.cast(lbl > 0.5, tf.uint8)
    return image, lbl


def circularidade(mask_comp: np.ndarray) -> float:
    area = float(mask_comp.sum())
    if area <= 0:
        return 0.0
    borda = np.logical_xor(mask_comp, ndi.binary_erosion(mask_comp))
    perimetro = float(borda.sum())
    if perimetro <= 0:
        return 0.0
    return float((4.0 * math.pi * area) / (perimetro * perimetro + 1e-6))


def limpar_geom(mask_bin: np.ndarray, min_area: int, max_area: int, min_circularidade: float) -> np.ndarray:
    mask = mask_bin.astype(bool)
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))

    labeled, n = ndi.label(mask)
    out = np.zeros_like(mask, dtype=np.uint8)
    for idx in range(1, n + 1):
        comp = labeled == idx
        area = int(comp.sum())
        if area < min_area or area > max_area:
            continue
        circ = circularidade(comp)
        if circ < min_circularidade:
            continue
        out[comp] = 1
    return out


def metricas(y_true: np.ndarray, y_pred: np.ndarray):
    tp = np.logical_and(y_pred == 1, y_true == 1).sum()
    fp = np.logical_and(y_pred == 1, y_true == 0).sum()
    fn = np.logical_and(y_pred == 0, y_true == 1).sum()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    return float(precision), float(recall), float(f1), float(iou)


pasta_base = garantir_drive_montado()
modelo_path = os.path.join(pasta_base, 'Modelo_UNet_Jussara_2025_FINAL_v2.keras')
if not os.path.exists(modelo_path):
    raise FileNotFoundError(f'Modelo não encontrado: {modelo_path}')

KERNEL_SIZE = 128
READ_SIZE = 129
INPUT_BANDS = ['R_1', 'NIR_1', 'NDVI_1', 'R_2', 'NIR_2', 'NDVI_2']
LABEL_BAND = 'label_chip'
BATCH_SIZE = int(os.environ.get('BATCH_SIZE_COMPARE', '8'))
N_BATCHES = int(os.environ.get('N_BATCHES_COMPARE', '20'))
MIN_AREA = int(os.environ.get('GEOM_MIN_AREA', '80'))
MAX_AREA = int(os.environ.get('GEOM_MAX_AREA', '25000'))
MIN_CIRC = float(os.environ.get('GEOM_MIN_CIRC', '0.35'))
PRECISION_TOL = float(os.environ.get('PRECISION_TOL', '0.02'))

thr_values = os.environ.get('THRESHOLDS_FINE', '0.28,0.29,0.30,0.31,0.32')
THRESHOLDS = [float(x.strip()) for x in thr_values.split(',') if x.strip()]

caminho_dados = localizar_tfrecord(pasta_base)
print(f'📂 Dados usados: {caminho_dados}')
print(f'🎚️ Thresholds avaliados: {THRESHOLDS}')
print(f'🧹 Pós geométrico: min_area={MIN_AREA}, max_area={MAX_AREA}, min_circularidade={MIN_CIRC}')

model = tf.keras.models.load_model(modelo_path, compile=False)

dataset = tf.data.TFRecordDataset(caminho_dados, compression_type='GZIP').map(
    parse_fast, num_parallel_calls=tf.data.AUTOTUNE
).batch(BATCH_SIZE).take(N_BATCHES)

all_probs, all_true = [], []
for x_batch, y_batch in dataset:
    probs = model.predict(x_batch, verbose=0)
    all_probs.append(probs)
    all_true.append(y_batch.numpy())

if not all_probs:
    raise RuntimeError('Nenhum batch disponível para avaliação.')

probs = np.concatenate(all_probs, axis=0)
y_true = np.concatenate(all_true, axis=0).astype(np.uint8)

linhas = []
for thr in THRESHOLDS:
    pred_raw = (probs > thr).astype(np.uint8)
    p, r, f1, iou = metricas(y_true, pred_raw)
    linhas.append({'modo': 'bruto', 'thr': thr, 'precision': p, 'recall': r, 'f1': f1, 'iou': iou})

    pred_geom = np.zeros_like(pred_raw, dtype=np.uint8)
    for i in range(pred_raw.shape[0]):
        pred_geom[i, :, :, 0] = limpar_geom(pred_raw[i, :, :, 0], MIN_AREA, MAX_AREA, MIN_CIRC)
    p2, r2, f12, iou2 = metricas(y_true, pred_geom)
    linhas.append({'modo': 'geom', 'thr': thr, 'precision': p2, 'recall': r2, 'f1': f12, 'iou': iou2})

brutos = [x for x in linhas if x['modo'] == 'bruto']
melhor_bruto = max(brutos, key=lambda x: x['f1'])
limite_prec = melhor_bruto['precision'] - PRECISION_TOL

candidatos = [x for x in linhas if x['precision'] >= limite_prec]
if candidatos:
    recomendado = max(candidatos, key=lambda x: (x['recall'], x['f1']))
else:
    recomendado = max(linhas, key=lambda x: x['f1'])

out_txt = os.path.join(pasta_base, 'resultado_recall_precision_fino.txt')
with open(out_txt, 'w', encoding='utf-8') as f:
    f.write('modo\tthr\tprecision\trecall\tf1\tiou\n')
    for row in linhas:
        f.write(
            f"{row['modo']}\t{row['thr']:.2f}\t{row['precision']:.6f}\t{row['recall']:.6f}\t{row['f1']:.6f}\t{row['iou']:.6f}\n"
        )
    f.write('\n')
    f.write(f"baseline_f1_thr={melhor_bruto['thr']:.2f}\n")
    f.write(f"baseline_f1_precision={melhor_bruto['precision']:.6f}\n")
    f.write(f"precision_floor={limite_prec:.6f}\n")
    f.write(
        f"recomendado=modo:{recomendado['modo']},thr:{recomendado['thr']:.2f},"
        f"precision:{recomendado['precision']:.6f},recall:{recomendado['recall']:.6f},"
        f"f1:{recomendado['f1']:.6f},iou:{recomendado['iou']:.6f}\n"
    )

print('📊 Resultado salvo em:', out_txt)
print('✅ Recomendado:', recomendado)
