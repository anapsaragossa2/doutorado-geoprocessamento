# doutorado-geoprocessamento

Repositório com experimentos de geoprocessamento para identificação de pivôs centrais em Jussara-GO.

## Detecção de pivôs centrais por forma no Google Earth Engine

O script [`gee/detectar_pivos_jussara_gee.js`](gee/detectar_pivos_jussara_gee.js) implementa um fluxo supervisionado para detectar pivôs centrais pelo padrão geométrico e textural, reduzindo a dependência exclusiva da assinatura espectral da cultura.

### Fluxo implementado

1. **Carregamento espacial e amostral**
   - Filtra a malha municipal do IBGE para Jussara-GO.
   - Carrega o asset de pivôs conhecidos como classe positiva.
   - Permite informar amostras negativas próprias ou gerar pontos aleatórios fora dos pivôs conhecidos.

2. **Mosaico Sentinel-2 de 2025**
   - Usa `COPERNICUS/S2_SR_HARMONIZED`.
   - Remove nuvens, sombras, cirrus e neve/gelo com `SCL` e `QA60`.
   - Cria uma mediana anual e índices NDVI/NDWI.

3. **Métricas de textura e formato**
   - Calcula textura GLCM (`contrast`, `corr`, `ent`) para NIR e NDVI.
   - Calcula desvio padrão espacial com kernels circulares de 3 e 7 pixels.
   - Essas camadas ajudam o Random Forest a aprender bordas circulares e transições abruptas típicas de pivôs.

4. **Classificação supervisionada**
   - Extrai os valores das bandas originais, índices e texturas nas amostras.
   - Divide amostras em treino/teste.
   - Treina `ee.Classifier.smileRandomForest()` e imprime matriz de confusão, acurácia, Kappa e importância das variáveis.

5. **Pós-processamento**
   - Classifica Jussara inteira.
   - Aplica filtros morfológicos (`focal_min` e `focal_max`) para reduzir ruídos isolados.
   - Exporta o raster final para o Google Drive.

### Como usar

1. Abra o [Google Earth Engine Code Editor](https://code.earthengine.google.com/).
2. Copie o conteúdo de `gee/detectar_pivos_jussara_gee.js`.
3. Ajuste no bloco `CONFIG`:
   - `assetPivosPositivos`: caminho do asset com pivôs conhecidos.
   - `assetAmostrasNegativas`: caminho opcional para amostras de não pivô.
   - `municipiosIbge`: caso a sua malha municipal esteja em outro asset.
4. Execute o script, valide as métricas impressas no console e ajuste as amostras caso haja confusão com áreas urbanas, bordas de lavouras ou corpos d'água.
5. Rode a tarefa `Export.image.toDrive` na aba **Tasks**.

### Inicialização em notebooks Python/Colab

Se você executar etapas do Earth Engine pela API Python, inicialize o `ee` com um projeto Google Cloud explícito. O erro `EEException: ee.Initialize: no project found` indica que `ee.Initialize()` foi chamado sem `project=`.

```python
import ee
from google.colab import userdata
from gee.earth_engine_colab_init import initialize_earth_engine

PROJECT_ID = userdata.get('EE_PROJECT_ID')  # ou use 'meu-projeto-gcp' diretamente
initialize_earth_engine(PROJECT_ID)
```

Antes de rodar o notebook, habilite o Earth Engine no projeto Google Cloud escolhido e, no Colab, salve o ID em **Secrets** com o nome `EE_PROJECT_ID` ou passe o valor diretamente para `initialize_earth_engine()`.


### Treinamento a partir de um ZIP no Colab

O script `train_pivos_from_zip.py` foi adicionado à raiz do repositório para evitar o erro `FileNotFoundError: train_pivos_from_zip.py não está no runtime do Colab`. O arquivo não aparece sozinho no Colab quando você abre apenas o `.ipynb`; por isso o notebook agora tem uma célula inicial que materializa `train_pivos_from_zip.py` no runtime. Alternativamente, clone o repositório ou faça upload do arquivo antes de chamar o treinamento.

Estrutura esperada do ZIP:

```text
images/nome_do_chip.png
masks/nome_do_chip.png
```

Os pares são associados pelo mesmo nome-base. Exemplo de execução no Colab após clonar o repositório:

```bash
python train_pivos_from_zip.py \
  --zip /content/drive/MyDrive/Tese_IA_Jussara/pivos_jussara.zip \
  --workdir /content/pivos_dataset \
  --image-size 128 \
  --batch-size 16 \
  --epochs 30 \
  --output /content/drive/MyDrive/Tese_IA_Jussara/Modelo_UNet_Pivos.keras \
  --overwrite
```

## Notebook de treinamento neural

O notebook `Script_Doutorando_v3_15_01_26.ipynb` contém scripts em Python/Colab para retreinamento e validação visual de um modelo U-Net salvo no Google Drive.
