# doutorado-geoprocessamento

Notebook de retreinamento e validação visual da U-Net para identificar pivôs de irrigação nos dados da tese.

## Safra 2025

O notebook `Script_Doutorando_v3_15_01_26.ipynb` está configurado para trabalhar com a safra de 2025 por meio da variável `SAFRA_ALVO = '2025'`.

### Como preparar os arquivos no Google Drive

1. Coloque os TFRecords da safra 2025 em `/content/drive/MyDrive/Tese_IA_Jussara`.
2. Garanta que o nome do arquivo contenha `2025` e `tfrecord`.
3. Se possível, mantenha também `MASSIVE` no nome do arquivo, pois o notebook prioriza esses arquivos.

Exemplos de nomes aceitos:

- `Jussara_2025_MASSIVE.tfrecord.gz`
- `MASSIVE_Jussara_2025.tfrecord`
- `Jussara_2025_treinamento.tfrecord.gz`

### O que o notebook faz para 2025

- Busca primeiro TFRecords `MASSIVE` da safra 2025.
- Impede o treinamento se não encontrar um TFRecord com `2025` no nome, evitando usar dados de outra safra por engano.
- Usa as bandas originais (`R`, `NIR` e `NDVI` em duas datas) e também atributos fenológicos derivados para ajudar o modelo a diferenciar plantio de vegetação nativa do Cerrado no período da safra.
- Salva checkpoints em `Modelo_Checkpoint_Jussara_2025.keras`.
- Salva o modelo final em `Modelo_UNet_Jussara_2025_FINAL_v2.keras`.
- Valida o modelo final usando TFRecords da mesma safra 2025.

### Datas usadas

A janela fenológica configurada no notebook é de **1º de dezembro de 2024 a 31 de março de 2025** (`SAFRA_INICIO = '2024-12-01'` e `SAFRA_FIM = '2025-03-31'`).

No TFRecord esperado, as bandas com sufixo `_1` representam a janela inicial da safra (dezembro/2024) e as bandas com sufixo `_2` representam a janela final da safra (março/2025). Portanto, a fenologia é calculada principalmente pela diferença entre `NDVI_2` e `NDVI_1`.

Importante: o notebook usa essas datas como configuração/documentação do experimento. Ele assume que o TFRecord foi gerado previamente com imagens desse período.

### Fenologia usada no treinamento

Com `USAR_FENOLOGIA = True`, o notebook adiciona seis bandas derivadas às seis bandas originais. Com `USAR_AUGE_VIGOR = True`, adiciona quatro bandas para mapear o auge do vigor. Com `USAR_VARIACAO_PAISAGEM = True`, adiciona mais seis bandas de variação da paisagem, totalizando 22 bandas de entrada:

- `DELTA_NDVI`: diferença de NDVI entre as duas datas, útil para capturar crescimento ou queda da lavoura.
- `ABS_DELTA_NDVI`: amplitude fenológica, útil porque vegetação nativa do Cerrado tende a ser mais estável que áreas plantadas em várias janelas da safra.
- `NDVI_MAX`: pico de vigor vegetativo observado no período.
- `NDVI_MEAN`: vigor médio no período analisado.
- `PLANTIO_SIGNAL`: aumento positivo de NDVI, usado como sinal de implantação/desenvolvimento do plantio.
- `CERRADO_STABILITY`: combinação de vigor médio com baixa variação temporal, usada como pista para vegetação nativa estável.

### Auge do vigor

Com apenas duas janelas temporais no TFRecord (`_1` para dezembro/2024 e `_2` para março/2025), o notebook mapeia o auge do vigor como o maior NDVI observado entre essas duas janelas. Se futuramente o TFRecord tiver mais datas intermediárias, esse cálculo deve ser expandido para procurar o máximo em toda a série temporal.

As bandas adicionadas são:

- `VIGOR_MAX_NDVI`: maior NDVI observado entre as duas janelas.
- `VIGOR_PEAK_TIMING`: indica se o pico ocorreu mais próximo da janela inicial (`0`) ou final (`1`).
- `VIGOR_PEAK_CONFIDENCE`: diferença absoluta entre `NDVI_1` e `NDVI_2`, indicando quão forte é a evidência do pico.
- `VIGOR_ALTO_MASK`: máscara de alto vigor usando `LIMIAR_ALTO_VIGOR_NDVI = 0.55`.

A validação imprime o percentual de pixels em alto vigor e mostra uma coluna visual `Auge do vigor`.

### Variação da paisagem e alertas

Além da fenologia, o notebook cria bandas para monitorar mudança da paisagem e gerar alertas simples no lote de validação:

- `COLHEITA_SIGNAL`: queda de NDVI entre a janela inicial e final, indicando possível colheita ou senescência.
- `SOLO_EXPOSTO_1`: provável solo exposto na janela inicial.
- `SOLO_EXPOSTO_2`: provável solo exposto na janela final.
- `SOLO_EXPOSTO_AUMENTO`: aumento de solo exposto entre as duas datas.
- `MUDANCA_PAISAGEM`: variação média em `R`, `NIR` e `NDVI`.
- `ALERTA_MUDANCA`: máscara de alerta quando há sinal forte de colheita, aumento de solo exposto ou mudança relevante da paisagem.

A célula de validação imprime percentuais de alerta, sinal médio de colheita e solo exposto, além de mostrar uma coluna visual de `Alerta paisagem` ao lado do gabarito e da predição da IA.

### Passos de execução

1. Abra o notebook no Google Colab.
2. Execute o bloco de retreinamento para gerar o checkpoint e o modelo final.
3. Execute o bloco de validação visual para comparar satélite, gabarito e predição da IA.
