# doutorado-geoprocessamento

Notebooks de geoprocessamento e treinamento de modelos U-Net para a tese.

## Notebooks disponíveis

- `Script_Doutorando_v3_15_01_26.ipynb`: fluxo original de retreinamento seguro com checkpoints e prova visual.
- `Script_Doutorando_Safrinha_2025.ipynb`: fluxo completo para a **safrinha de 2025**. Primeiro cria/exporta amostras TFRecord próprias da safrinha no Earth Engine, na versão `PATCHES_V3_FENO` para incluir métricas fenológicas e garantir `label_chip` como patch 129×129 (`AMOSTRAS_SAFRINHA_2025_*.tfrecord.gz`), já configurado com `EE_PROJECT_ID = 'sefazgogeoprocessamento'` e o asset `projects/sefazgogeoprocessamento/assets/Gabarito_Milho_UPLOAD`; para município inteiro, tenta buscar automaticamente o limite de Jussara-GO e começa em modo de teste rápido com exportação dividida em partes menores, e, se o TFRecord ainda não existir, inicia a exportação automaticamente, tenta aguardar a task finalizar para continuar o treino na mesma execução e só pausa se o arquivo ainda não estiver disponível; inclui uma análise espectral/fenológica do milho por bandas e índices Sentinel-2, depois retreina a U-Net e faz a prova visual usando somente essas amostras, evitando reutilizar TFRecords da safra principal.
