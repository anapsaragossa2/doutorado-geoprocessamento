"""Fluxo Colab + Google Earth Engine para gerar recortes de Jussara-GO.

Este arquivo foi escrito no formato de células (`# %%`) para ser colado no
Google Colab ou aberto por editores compatíveis. Ele usa o GEE para recortar
Jussara-GO, exporta um mosaico Sentinel-2 para o Google Drive e, depois, chama o
script local de treinamento com o ZIP de máscaras/imagens de pivôs.
"""

# %% [markdown]
# # Identificação de pivôs em Jussara-GO com Colab + Google Earth Engine
#
# 1. Autentique o Google Earth Engine.
# 2. Gere o recorte de Jussara-GO no GEE.
# 3. Exporte um mosaico Sentinel-2 com bandas úteis para o Drive.
# 4. Treine a U-Net usando `projects/sefazgogeoprocessamento/assets/pivo.zip`.

# %%
# No Colab, descomente se precisar instalar dependências:
# !pip install -q earthengine-api geemap tensorflow

# %%
from pathlib import Path

import ee

PROJECT_ID = None  # opcional: informe o ID do projeto GEE/Google Cloud, se exigido pela conta.
DRIVE_FOLDER = "gee_jussara_pivos"
ZIP_PIVOS = "projects/sefazgogeoprocessamento/assets/pivo.zip"

# %%
ee.Authenticate()
if PROJECT_ID:
    ee.Initialize(project=PROJECT_ID)
else:
    ee.Initialize()

# %%
def jussara_go_geometry() -> ee.Geometry:
    """Retorna o limite municipal de Jussara-GO usando a base GAUL nível 2."""
    municipios = ee.FeatureCollection("FAO/GAUL/2015/level2")
    jussara = (
        municipios.filter(ee.Filter.eq("ADM0_NAME", "Brazil"))
        .filter(ee.Filter.eq("ADM1_NAME", "Goias"))
        .filter(ee.Filter.eq("ADM2_NAME", "Jussara"))
    )
    return jussara.geometry()


def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Remove nuvens e cirrus usando a banda QA60 do Sentinel-2 SR Harmonized."""
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000)


def sentinel2_jussara_composite(start_date: str, end_date: str) -> ee.Image:
    """Cria mosaico Sentinel-2 recortado para Jussara-GO com NDVI e NDWI."""
    aoi = jussara_go_geometry()
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 35))
        .map(mask_s2_clouds)
    )
    composite = collection.median().clip(aoi)
    ndvi = composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = composite.normalizedDifference(["B3", "B8"]).rename("NDWI")
    return composite.select(["B2", "B3", "B4", "B8", "B11", "B12"]).addBands([ndvi, ndwi])

# %%
aoi = jussara_go_geometry()
image = sentinel2_jussara_composite("2024-05-01", "2024-10-31")

export_task = ee.batch.Export.image.toDrive(
    image=image,
    description="jussara_go_sentinel2_pivos_2024",
    folder=DRIVE_FOLDER,
    fileNamePrefix="jussara_go_sentinel2_pivos_2024",
    region=aoi,
    scale=10,
    maxPixels=1e13,
)
export_task.start()
print("Exportação iniciada. Acompanhe em: https://code.earthengine.google.com/tasks")
print("Task ID:", export_task.id)

# %% [markdown]
# Depois que a exportação terminar, o GeoTIFF estará no Google Drive. O treinamento
# abaixo usa o ZIP `pivo.zip` com os pares imagem/máscara. Se o ZIP estiver no Drive,
# monte o Drive e ajuste `ZIP_PIVOS`.

# %%
# from google.colab import drive
# drive.mount('/content/drive')
# ZIP_PIVOS = '/content/drive/MyDrive/caminho/para/pivo.zip'

# %%
# Rode o treinamento no Colab após confirmar que o ZIP existe.
# Se este arquivo estiver na raiz do repositório clonado, o comando abaixo funciona:
# !python train_pivos_from_zip.py --zip "$ZIP_PIVOS" --epochs 40 --batch-size 8 --output Modelo_UNet_Pivos_Jussara.keras

print(f"ZIP configurado para treinamento: {Path(ZIP_PIVOS)}")
