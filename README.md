# doutorado-geoprocessamento

## Treinamento para identificação de pivôs em Jussara-GO

O fluxo recomendado agora é **Google Colab + Google Earth Engine (GEE)**:

1. usar o GEE para definir o recorte municipal de **Jussara-GO**;
2. exportar um mosaico Sentinel-2 recortado para o Google Drive;
3. treinar a U-Net com o ZIP de pivôs em `projects/sefazgogeoprocessamento/assets/pivo.zip`.

Para abrir diretamente no Colab, use o notebook `colab_gee_jussara_pivos.ipynb`.
Também mantive `colab_gee_jussara_pivos.py` em células `# %%` para quem prefere
versionar/editar como script.

### Recorte GEE de Jussara-GO

O script usa a coleção `FAO/GAUL/2015/level2` e filtra:

- `ADM0_NAME = Brazil`
- `ADM1_NAME = Goias`
- `ADM2_NAME = Jussara`

Depois monta um mosaico `COPERNICUS/S2_SR_HARMONIZED` com bandas `B2`, `B3`,
`B4`, `B8`, `B11`, `B12`, `NDVI` e `NDWI`, recortado para o município.

### Como executar no Colab

No Colab, instale dependências se necessário:

```python
!pip install -q earthengine-api geemap tensorflow
```

Abra `colab_gee_jussara_pivos.ipynb` no Colab e execute as células em ordem.
Se preferir usar o script, rode:

```python
%run colab_gee_jussara_pivos.py
```

A exportação será enviada para a pasta `gee_jussara_pivos` do Google Drive. Em
seguida, ajuste o caminho do ZIP se ele estiver no Drive e rode:

```bash
python train_pivos_from_zip.py \
  --zip projects/sefazgogeoprocessamento/assets/pivo.zip \
  --epochs 40 \
  --batch-size 8 \
  --output Modelo_UNet_Pivos_Jussara.keras
```

### Estrutura esperada do ZIP de pivôs

O script de treinamento tenta parear automaticamente imagens e máscaras pelo
nome do arquivo. A forma recomendada é:

```text
pivo.zip
├── images/
│   ├── area_001.png
│   └── area_002.png
└── masks/
    ├── area_001.png
    └── area_002.png
```

Também são aceitos nomes de pastas/arquivos com termos como `image`, `img`,
`imagem`, `mask`, `label`, `mascara` e `pivo`.

### Parâmetros úteis do treinamento

- `--image-size`: tamanho quadrado para redimensionar imagens e máscaras (padrão `128`).
- `--channels`: número de canais das imagens de entrada (padrão `3`).
- `--extract-dir`: diretório opcional para manter os arquivos extraídos para inspeção.
- `--patience`: paciência do `EarlyStopping` (padrão `8`).

O modelo final será salvo no arquivo definido por `--output`.
