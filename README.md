# doutorado-geoprocessamento

Scripts de apoio para retreinamento e validação visual de uma U-Net para segmentação de pivôs.

## Arquivos principais

- `treinamento_seguro_unet.py`: pipeline de retreinamento com checkpoints, retomada automática e callbacks de segurança.
- `validacao_visual_modelo.py`: prova de fogo para carregar o modelo salvo e visualizar predições vs. gabarito.
- `treinamento_unet_colab.ipynb`: notebook pronto para abrir no Colab e executar treino + validação.

## Não aparece no Colab (GitHub)?

No seletor do Colab, ele lista principalmente arquivos **`.ipynb`**. Como antes só tinha `.py`, podia aparecer “Nenhum resultado”.

Agora use o notebook:

- `treinamento_unet_colab.ipynb`

Dica no Colab:
1. Em **Abrir notebook > GitHub**, selecione o repositório.
2. Troque para a branch correta.
3. Procure por `treinamento_unet_colab.ipynb`.

## Em qual pasta o modelo é salvo?

Agora o padrão está configurado para salvar **aqui no projeto**:

`/workspace/doutorado-geoprocessamento`

Arquivos gerados no treino:

- `Modelo_Checkpoint_last.keras`
- `Modelo_Checkpoint_best.keras`
- `historico_treinamento.csv`
- `Modelo_UNet_Jussara_2025_FINAL_v2.keras`

Se quiser trocar a pasta base, use a variável de ambiente `TESE_IA_BASE_DIR`.

Exemplo:

```bash
export TESE_IA_BASE_DIR="/caminho/que/voce/quiser"
python treinamento_seguro_unet.py
```

## Uso

1. Deixe o `.tfrecord` dentro da pasta base (padrão: `/workspace/doutorado-geoprocessamento`).
2. Execute o retreinamento:

```bash
python treinamento_seguro_unet.py
```

3. Depois execute a validação visual:

```bash
python validacao_visual_modelo.py
```
