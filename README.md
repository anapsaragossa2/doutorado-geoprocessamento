# doutorado-geoprocessamento

Código para retreinamento/validação da U-Net no Colab com foco em identificar talhões com mais rigor.

## Fluxo único no mesmo notebook (sequencial)

Os dois notebooks estão com o pipeline completo:

- `treinamento_unet_colab.ipynb`
- `Script_Doutorando_v3_15_01_26.ipynb`

Ordem principal no notebook:
1. Treino
2. Prova real visual
3. **Pós-processamento da máscara** (limpeza)
4. Sweep grosso de limiares
5. **Sweep fino automático** ao redor do melhor limiar
6. Resumo final (tabela + gráfico + threshold recomendado)

## Resultado mais recente de sweep fino

Valores reportados:

- `thr=0.30` → `precision=0.716225`, `recall=0.623033`, `f1=0.666386`, `iou=0.499685`
- `thr=0.32` → `precision=0.724329`, `recall=0.616820`, `f1=0.666265`, `iou=0.499549`
- `thr=0.34` → `precision=0.732084`, `recall=0.611038`, `f1=0.666106`, `iou=0.499370`
- `thr=0.36` → `precision=0.739467`, `recall=0.605537`, `f1=0.665833`, `iou=0.499064`

Conclusão atual:

- melhor limiar por F1: **`0.30`**
- defaults de inferência foram alinhados para `THRESHOLD_INFERENCIA = 0.30`

## Novos blocos adicionados

### 1) Pós-processamento (limpeza de máscara)
Script: `pos_processamento_mascara.py`

- aplica abertura/fechamento morfológico;
- remove componentes pequenos (`MIN_COMPONENT_SIZE`);
- compara métricas antes/depois (`precision`, `recall`, `f1`, `iou`);
- salva imagem comparativa em:
  - `/content/drive/MyDrive/Tese_IA_Jussara/comparativo_pos_processamento.png`

### 2) Sweep fino automático
Script base: `teste_limiares_talhoes.py`

- faz sweep grosso e detecta `melhor_thr`;
- gera automaticamente um sweep fino ao redor do melhor valor;
- salva dois relatórios:
  - `/content/drive/MyDrive/Tese_IA_Jussara/resultado_teste_limiares.txt`
  - `/content/drive/MyDrive/Tese_IA_Jussara/resultado_teste_limiares_fino.txt`

## Scripts Python

- `treinamento_seguro_unet.py`
- `validacao_visual_modelo.py`
- `teste_limiares_talhoes.py`
- `pos_processamento_mascara.py`
