# doutorado-geoprocessamento

Código restaurado para o formato original de retreinamento/validação no Colab.

## Scripts

- `treinamento_seguro_unet.py`
- `validacao_visual_modelo.py`
- `treinamento_unet_colab.ipynb`

## Erro comum no Colab (repo privado)

Se o clone falhar no Colab, normalmente é porque o repositório está **privado**.
Nesse caso, no notebook `treinamento_unet_colab.ipynb` preencha:

- `GITHUB_USER`
- `REPO_NAME`
- `REPO_BRANCH`
- `GITHUB_TOKEN` (token com permissão de leitura)

Também confira se a branch escolhida realmente contém os arquivos:

- `treinamento_seguro_unet.py`
- `validacao_visual_modelo.py`

## Pasta padrão (como no código original)

`/content/drive/MyDrive/Tese_IA_Jussara`

> Rode no Google Colab com o Drive montado.
