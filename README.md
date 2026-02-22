# doutorado-geoprocessamento

Código restaurado para o formato original de retreinamento/validação no Colab.

## Scripts

- `treinamento_seguro_unet.py`
- `validacao_visual_modelo.py`
- `treinamento_unet_colab.ipynb`
- `Script_Doutorando_v3_15_01_26.ipynb` ✅ (integrado ao fluxo atual)

## Erro comum no Colab (repo privado)

Se aparecer erro como:

- `fatal: could not read Username for 'https://github.com': No such device or address`
- `remote: Write access to repository not granted.`
- `The requested URL returned error: 403`

os notebooks agora tentam clone público primeiro e, se necessário, pedem:

- `GITHUB_AUTH_USER` (usuário que **gerou o token**)
- `GITHUB_TOKEN`

### Checklist para erro 403

- Se token classic: habilite escopo `repo`.
- Se fine-grained: acesso ao repositório + permissão `Contents: Read-only`.
- Se organização usa SSO: autorize o token para a organização.
- Confirme `GITHUB_OWNER`, `REPO_NAME` e `REPO_BRANCH`.

## Pasta padrão (como no código original)

`/content/drive/MyDrive/Tese_IA_Jussara`

> Rode no Google Colab com o Drive montado.
