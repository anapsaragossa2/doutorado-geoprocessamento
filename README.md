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

4. **Classificação adaptativa**
   - Extrai os valores das bandas originais, índices e texturas nas amostras.
   - Divide amostras em treino/teste.
   - Com amostras positivas, treina `ee.Classifier.smileRandomForest()` e imprime matriz de confusão, acurácia, Kappa e importância das variáveis.
   - Sem amostras, usa bordas Canny e estatísticas em um kernel circular para produzir candidatos sem interromper a execução.

5. **Pós-processamento**
   - Classifica Jussara inteira.
   - Aplica filtros morfológicos (`focal_min` e `focal_max`) para reduzir ruídos isolados.
   - Exporta o raster final para o Google Drive.

### Como usar

1. Abra o [Google Earth Engine Code Editor](https://code.earthengine.google.com/).
2. Copie o conteúdo de `gee/detectar_pivos_jussara_gee.js`.
3. Ajuste no bloco `CONFIG`:
   - `assetAmostrasRotuladas`: opção recomendada para a coleção criada pelo painel de marcação, contendo `classe = 1` e `classe = 0`.
   - `assetPivosPositivos`: já está configurado para o SHP de pivôs fornecido,
     `projects/sefazgogeoprocessamento/assets/final_pivos3`. Não use literalmente
     `users/SEU_USUARIO/...`.
   - Se não tiver um asset, preencha `AMOSTRAS_POSITIVAS_INLINE` com geometrias desenhadas no Code Editor e altere `usarAmostrasPositivasInline` para `true`.
   - `assetAmostrasNegativas`: caminho opcional para amostras de não pivô.
   - O limite de Jussara é carregado por padrão da coleção pública `FAO/GAUL/2015/level2`; ajuste os campos apenas se trocar essa coleção.
4. Execute o script, valide as métricas impressas no console e ajuste as amostras caso haja confusão com áreas urbanas, bordas de lavouras ou corpos d'água.
5. Rode a tarefa `Export.image.toDrive` na aba **Tasks**.

#### Erro `Computed value is too large`

O detector limita automaticamente o treinamento a 2.000 pontos de pivô e 2.000
pontos de não pivô. Isso é importante porque usar `sampleRegions` diretamente em
todos os pixels de muitos polígonos do SHP pode exceder o limite de cálculo do
Earth Engine e impedir a matriz de confusão e a camada classificada de serem
geradas. Se ainda for necessário reduzir o processamento, diminua
`quantidadePontosPositivos` e `quantidadePontosNegativos` no `CONFIG` (por
exemplo, para `1000`); não aumente esses valores antes de confirmar que o fluxo
executa normalmente.

### Usar o SHP para pivôs e marcar somente não pivôs no GEE

O script [`gee/marcar_amostras_pivos_gee.js`](gee/marcar_amostras_pivos_gee.js)
abre um painel de rotulagem no Code Editor. Nesse fluxo, os pivôs positivos vêm
do Asset `projects/sefazgogeoprocessamento/assets/final_pivos3`; no mapa você desenha somente áreas negativas
e usa o botão **Adicionar como NÃO PIVÔ** (`classe = 0`). Ao terminar, o painel
combina os polígonos do SHP (`classe = 1`) com os desenhos (`classe = 0`) e exporta
uma única coleção para Asset ou Drive.
Internamente, a camada de desenho usa o nome alfanumérico `DesenhoAtual`, pois o
GEE não aceita espaços ou caracteres acentuados no nome de uma `GeometryLayer`.
O campo **Asset ID de saída** precisa ser preenchido antes da exportação para
Asset; se ficar vazio, o painel mostra a orientação sem encerrar o aplicativo.

Para um treinamento útil, distribua os não pivôs pelo município e inclua pastagens,
matas, área urbana, rios, lavouras retangulares e círculos que não sejam pivôs.
Evite polígonos atravessando a borda entre duas classes e procure produzir uma
quantidade de áreas negativas compatível com a diversidade do SHP positivo.

Depois da exportação para Asset, copie o caminho gerado para
`CONFIG.assetAmostrasRotuladas` em `detectar_pivos_jussara_gee.js`. O detector
separa automaticamente `classe = 1` e `classe = 0` e executa o Random Forest.

### Usar o SHP `final_pivos3` como treinamento

Se a marcação manual não produzir bons resultados, compacte juntos os arquivos
do shapefile (`.shp`, `.shx`, `.dbf` e `.prj`) e envie o ZIP pela aba **Assets**
do Earth Engine. Depois, importe a tabela no Code Editor e altere o nome da
variável importada para `final_pivos3` — com sublinhado e sem espaço.

O detector reconhece automaticamente essa variável e trata todos os polígonos do
SHP como `classe = 1` (pivô). Não é necessário preencher
`assetPivosPositivos`. Na ausência de uma coleção negativa, o script gera pontos
de `classe = 0` fora dos pivôs e do buffer de 250 metros. Se preferir usar o
Asset sem importá-lo pela interface, copie seu ID completo, no formato
`projects/SEU_PROJETO/assets/final_pivos3`, para `assetPivosPositivos`.

Antes do treinamento, confira se os polígonos aparecem na camada **Amostras
positivas**, se estão dentro de Jussara e se representam apenas pivôs. O nome
visível do Asset pode ser `final_pivos3`, mas o código sempre precisa da variável
importada ou do ID completo — apenas o texto curto não é um endereço de Asset.

O erro `Collection.loadTable: Collection asset 'users/SEU_USUARIO/...' not
found` significa que um texto de exemplo foi executado como se fosse um asset.
O script agora usa `null` como padrão. Sem amostras positivas, ele não interrompe
a execução: muda automaticamente para uma detecção não supervisionada baseada em
bordas Canny, densidade de bordas e variação de NDVI dentro de uma janela circular.
Nesse modo não existe matriz de confusão, pois não há rótulos de referência. Ao
informar amostras positivas, o fluxo volta automaticamente ao Random Forest.
`Map.centerObject(jussara, 11)` não é a origem do erro; ele apenas centraliza a
visualização depois que os dados e o modelo foram preparados.

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

O script `train_pivos_from_zip.py` está na raiz do repositório, mas o notebook não depende mais desse arquivo para iniciar o treinamento. A primeira célula carrega a implementação embutida diretamente na função `train_from_zip`; a última célula chama essa função sem testar `Path('train_pivos_from_zip.py').exists()` e sem lançar o `FileNotFoundError` relatado. A cópia `.py` criada no runtime é apenas uma conveniência para uso via linha de comando.

> **Importante:** a janela **Arquivo > Abrir notebook > GitHub** do Colab não é um
> navegador de arquivos do repositório. Ela lista **somente arquivos `.ipynb`**.
> Portanto, é esperado que `README.md`, `train_pivos_from_zip.py` e a pasta `gee/`
> não apareçam nessa tela, mesmo estando corretamente versionados na mesma branch.

Para levar todos os arquivos da branch ao runtime, abra um terminal/célula no Colab
e clone o repositório. Troque `NOME_DA_BRANCH` pela branch selecionada na janela do
Colab (por exemplo, `codex/treinar-identificacao-de-pivos`):

```bash
!git clone --branch NOME_DA_BRANCH --single-branch \
  https://github.com/anapsaragossa2/doutorado-geoprocessamento.git \
  /content/doutorado-geoprocessamento
%cd /content/doutorado-geoprocessamento
!git ls-files
```

Em repositório particular, o `git clone` exige autenticação do GitHub. Nesse caso,
a célula inicial do notebook continua sendo a opção mais simples: ela cria uma
cópia autocontida de `train_pivos_from_zip.py` sem depender do clone. Depois de
executá-la, confirme o arquivo no runtime com `!ls -l train_pivos_from_zip.py`.

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

Os notebooks `Script_Doutorando_v3_15_01_26.ipynb` e
`colab_gee_jussara_pivos.ipynb` contêm o mesmo fluxo autocontido para
retreinamento e validação visual de um modelo U-Net salvo no Google Drive. A
segunda cópia existe porque esse é o nome exibido no seletor do Colab usado no
fluxo GEE.

Se o traceback ainda mostrar literalmente o bloco
`if not Path('train_pivos_from_zip.py').exists():`, o Colab está executando uma
revisão antiga que ficou aberta no navegador: esse bloco não existe nas células
de código dos notebooks atuais. Feche a guia, abra novamente o notebook na
branch atual pelo GitHub e escolha **Ambiente de execução > Reiniciar sessão**.
Depois execute a primeira célula antes da célula final de treinamento.
