# doutorado-geoprocessamento

Scripts de apoio para retreinamento e validação visual de uma U-Net para segmentação de pivôs.

## Arquivos principais

- `treinamento_seguro_unet.py`: pipeline de retreinamento com checkpoints, retomada automática e callbacks de segurança.
- `validacao_visual_modelo.py`: prova de fogo para carregar o modelo salvo e visualizar predições vs. gabarito.
- `treinamento_unet_colab.ipynb`: notebook pronto para abrir no Colab e executar treino + validação.

## Abrir no Colab (GitHub)

Use estes comandos no notebook:

```python
# 1) Clonar o repositório (ajuste para seu usuário/repositorio)
!git clone https://github.com/Anapsaragossa/doutorado-geoprocessamento.git
%cd doutorado-geoprocessamento
```

Se der erro porque a pasta já existe, use a versão robusta do notebook (`1.b`).

## Em qual pasta o modelo é salvo?

Por padrão, os scripts salvam no diretório definido por `TESE_IA_BASE_DIR`.
Se não definir, usam:

`/workspace/doutorado-geoprocessamento`

Arquivos gerados no treino:

- `Modelo_Checkpoint_last.keras`
- `Modelo_Checkpoint_best.keras`
- `historico_treinamento.csv`
- `Modelo_UNet_Jussara_2025_FINAL_v2.keras`

Exemplo para trocar a pasta base:

```bash
export TESE_IA_BASE_DIR="/caminho/que/voce/quiser"
python treinamento_seguro_unet.py
```

## Uso

1. Deixe o `.tfrecord` dentro da pasta base.
2. Execute o retreinamento:

```bash
python treinamento_seguro_unet.py
```

3. Depois execute a validação visual:

```bash
python validacao_visual_modelo.py
```
