# doutorado-geoprocessamento

Notebooks de geoprocessamento e treinamento de modelos U-Net para a tese.

## Notebooks disponíveis

- `Script_Doutorando_v3_15_01_26.ipynb`: fluxo original de retreinamento seguro com checkpoints e prova visual.
- `Script_Doutorando_Safrinha_2025.ipynb`: fluxo equivalente adaptado para a **safrinha de 2025**, com busca específica por TFRecords da safrinha e nomes de checkpoint/modelo final isolados para evitar sobrescrever artefatos anteriores. Caso o TFRecord não tenha `SAFRINHA`/`2025` no nome, o notebook permite informar o caminho manualmente ou usar fallback genérico para o arquivo mais recente.
