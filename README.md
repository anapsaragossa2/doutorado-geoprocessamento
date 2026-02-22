# doutorado-geoprocessamento

Código para retreinamento/validação da U-Net no Colab com foco em identificar talhões com mais rigor.

## Fluxo recomendado (sem clone)

Use os notebooks abaixo. Eles gravam os scripts localmente e executam treino + validação no Colab:

- `treinamento_unet_colab.ipynb`
- `Script_Doutorando_v3_15_01_26.ipynb`

## O que ficou mais rigoroso no treino

- Normalização por banda em cada chip.
- Data augmentation (flip horizontal/vertical + rotação 90º).
- U-Net com BatchNormalization em todos os blocos.
- Loss combinada `BCE + Dice` para lidar melhor com desbalanceamento.
- Métricas: `accuracy`, `precision`, `recall`, `IoU`, `dice_coef`.
- Checkpoint do último e do melhor modelo (`val_iou`), `EarlyStopping`, `ReduceLROnPlateau` e `CSVLogger`.

## Scripts Python

- `treinamento_seguro_unet.py`
- `validacao_visual_modelo.py`

## Pasta padrão

`/content/drive/MyDrive/Tese_IA_Jussara`

> Execute no Google Colab com o Drive montado.
