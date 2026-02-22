# doutorado-geoprocessamento

Código para retreinamento/validação da U-Net no Colab com foco em identificar talhões com mais rigor.

## Fluxo único no mesmo notebook (sequencial)

Os dois notebooks agora estão organizados com **todos os testes em sequência**:

- `treinamento_unet_colab.ipynb`
- `Script_Doutorando_v3_15_01_26.ipynb`

Ordem no notebook:
1. Instala dependências
2. Monta Drive
3. Treino
4. Gráfico de evolução do treino (loss / IoU / Dice)
5. Prova real visual
6. Teste de limiares (threshold sweep)
7. Resumo final do threshold

Assim você acompanha toda a evolução no mesmo arquivo.

## Artefatos gerados no Drive

- Modelo final: `/content/drive/MyDrive/Tese_IA_Jussara/Modelo_UNet_Jussara_2025_FINAL_v2.keras`
- Checkpoint best: `/content/drive/MyDrive/Tese_IA_Jussara/Modelo_Checkpoint_best.keras`
- Histórico treino: `/content/drive/MyDrive/Tese_IA_Jussara/historico_treinamento.csv`
- Prova real: `/content/drive/MyDrive/Tese_IA_Jussara/prova_real_validacao.png`
- Sweep de limiares: `/content/drive/MyDrive/Tese_IA_Jussara/resultado_teste_limiares.txt`

## Scripts Python

- `treinamento_seguro_unet.py`
- `validacao_visual_modelo.py`
- `teste_limiares_talhoes.py`
