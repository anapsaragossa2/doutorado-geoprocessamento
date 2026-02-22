# doutorado-geoprocessamento

Scripts de apoio para retreinamento e validação visual de uma U-Net para segmentação de pivôs.

## Arquivos principais

- `treinamento_seguro_unet.py`: pipeline de retreinamento com checkpoints, retomada automática e callbacks de segurança.
- `validacao_visual_modelo.py`: prova de fogo para carregar o modelo salvo e visualizar predições vs. gabarito.

## Em qual pasta o modelo é salvo?

Por padrão, os scripts salvam e leem arquivos em:

`/content/drive/MyDrive/Tese_IA_Jussara`

Arquivos gerados no treino:

- `Modelo_Checkpoint_last.keras`
- `Modelo_Checkpoint_best.keras`
- `historico_treinamento.csv`
- `Modelo_UNet_Jussara_2025_FINAL_v2.keras`

Você pode mudar a pasta base definindo a variável de ambiente `TESE_IA_BASE_DIR`.

Exemplo:

```bash
export TESE_IA_BASE_DIR="/content/drive/MyDrive/Tese_IA_Jussara"
python treinamento_seguro_unet.py
```

## Uso no Google Colab

1. Suba os scripts no notebook/ambiente.
2. Execute o retreinamento:

```bash
python treinamento_seguro_unet.py
```

3. Depois execute a validação visual:

```bash
python validacao_visual_modelo.py
```
