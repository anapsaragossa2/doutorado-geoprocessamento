# doutorado-geoprocessamento

## Correção de separador decimal/colunas numéricas no CSV do GEE

Se a tabela exportada do Earth Engine estiver com colunas numéricas como texto (por causa de separador `,`/`;` ou decimal com vírgula), use:

```bash
python fix_colheita_csv.py --input colheita.csv --output colheita_corrigida.xlsx
```

O script:
- detecta automaticamente o separador (`;`, `,` ou tab);
- converte colunas numéricas para `float` com suporte a formatos `1.234,56` e `1,234.56`;
- recalcula `percentual_colhido_recalc` para conferência.
