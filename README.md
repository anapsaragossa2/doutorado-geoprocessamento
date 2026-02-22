# doutorado-geoprocessamento

Scripts de apoio para retreinamento e validação visual de uma U-Net para segmentação de pivôs.

## Arquivos principais

- `treinamento_seguro_unet.py`: pipeline de retreinamento com checkpoints, retomada automática e callbacks de segurança.
- `validacao_visual_modelo.py`: prova de fogo para carregar o modelo salvo e visualizar predições vs. gabarito.

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

> Ambos os scripts assumem os dados em `/content/drive/MyDrive/Tese_IA_Jussara`.
