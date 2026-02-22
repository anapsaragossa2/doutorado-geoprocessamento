# doutorado-geoprocessamento

Código restaurado para o formato original de retreinamento/validação no Colab.

## Scripts

- `treinamento_seguro_unet.py`
- `validacao_visual_modelo.py`
- `treinamento_unet_colab.ipynb`
- `Script_Doutorando_v3_15_01_26.ipynb` ✅ (integrado ao fluxo atual)

## Erro comum no Colab (repo privado)

Se aparecer erro como:

`fatal: could not read Username for 'https://github.com': No such device or address`

o notebook agora tenta:

1. clone público primeiro;
2. se detectar erro de autenticação, solicita token via `getpass()` e tenta novamente.

Você só precisa conferir:

- `GITHUB_USER`
- `REPO_NAME`
- `REPO_BRANCH`

E informar token do GitHub (read access) quando solicitado.

## Pasta padrão (como no código original)

`/content/drive/MyDrive/Tese_IA_Jussara`

> Rode no Google Colab com o Drive montado.
