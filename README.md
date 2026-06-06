# doutorado-geoprocessamento

Notebooks de geoprocessamento e treinamento de modelos U-Net para a tese.

## Notebooks disponíveis

- `Script_Doutorando_v3_15_01_26.ipynb`: fluxo original de retreinamento seguro com checkpoints e prova visual.
- `Script_Doutorando_Safrinha_2025.ipynb`: fluxo completo para a **safrinha de 2025**. Primeiro cria/exporta amostras TFRecord próprias da safrinha no Earth Engine (`AMOSTRAS_SAFRINHA_2025_*.tfrecord.gz`), já configurado com `EE_PROJECT_ID = 'sefazgogeoprocessamento'` e o asset `projects/sefazgogeoprocessamento/assets/Gabarito_Milho_UPLOAD`, depois retreina a U-Net e faz a prova visual usando somente essas amostras, evitando reutilizar TFRecords da safra principal.
