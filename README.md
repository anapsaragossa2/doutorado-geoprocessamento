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
- Salva checkpoints em `Modelo_Checkpoint_Jussara_2025.keras`.
- Salva o modelo final em `Modelo_UNet_Jussara_2025_FINAL_v2.keras`.
- Valida o modelo final usando TFRecords da mesma safra 2025.

### Passos de execução

1. Abra o notebook no Google Colab.
2. Execute o bloco de retreinamento para gerar o checkpoint e o modelo final.
3. Execute o bloco de validação visual para comparar satélite, gabarito e predição da IA.
