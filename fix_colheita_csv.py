#!/usr/bin/env python3
"""Normaliza CSV exportado do Google Earth Engine para cálculo de área colhida.

Uso:
    python fix_colheita_csv.py --input colheita.csv --output colheita_corrigida.xlsx
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

COLUNAS_NUMERICAS_PADRAO = [
    "area_ha",
    "area_ha_original",
    "area_total_calc_ha",
    "area_colhida_ha",
    "percentual_colhido",
]


def detectar_separador(caminho_csv: Path) -> str:
    """Detecta automaticamente separador do CSV (, ; ou tab)."""
    with caminho_csv.open("r", encoding="utf-8-sig", newline="") as f:
        amostra = f.read(4096)
    try:
        dialeto = csv.Sniffer().sniff(amostra, delimiters=",;\t")
        return dialeto.delimiter
    except csv.Error:
        return ","


def para_float_robusto(valor: object) -> float | None:
    """Converte número com variações de locale (1.234,56 / 1,234.56 / 1234.56)."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return None

    s = str(valor).strip()
    if s == "":
        return None

    # remove espaços e símbolos comuns
    s = s.replace(" ", "").replace("R$", "")

    if "," in s and "." in s:
        # Assume último símbolo como separador decimal
        if s.rfind(",") > s.rfind("."):
            # 1.234,56 -> 1234.56
            s = s.replace(".", "").replace(",", ".")
        else:
            # 1,234.56 -> 1234.56
            s = s.replace(",", "")
    elif "," in s:
        # 1234,56 -> 1234.56
        s = s.replace(",", ".")

    try:
        return float(s)
    except ValueError:
        return None


def normalizar_dataframe(df: pd.DataFrame, colunas_numericas: list[str]) -> pd.DataFrame:
    for coluna in colunas_numericas:
        if coluna in df.columns:
            df[coluna] = df[coluna].apply(para_float_robusto)

    # recalcula percentual como validação da consistência
    if {"area_colhida_ha", "area_total_calc_ha"}.issubset(df.columns):
        denom = df["area_total_calc_ha"].replace({0: pd.NA})
        df["percentual_colhido_recalc"] = (df["area_colhida_ha"] / denom) * 100

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Caminho do CSV exportado do GEE")
    parser.add_argument("--output", required=True, help="Caminho do arquivo de saída (.xlsx ou .csv)")
    parser.add_argument(
        "--colunas-numericas",
        nargs="*",
        default=COLUNAS_NUMERICAS_PADRAO,
        help="Lista de colunas numéricas para normalizar",
    )
    args = parser.parse_args()

    caminho_entrada = Path(args.input)
    caminho_saida = Path(args.output)

    separador = detectar_separador(caminho_entrada)
    df = pd.read_csv(caminho_entrada, sep=separador, dtype=str, encoding="utf-8-sig")
    df = normalizar_dataframe(df, args.colunas_numericas)

    if caminho_saida.suffix.lower() == ".xlsx":
        df.to_excel(caminho_saida, index=False)
    else:
        df.to_csv(caminho_saida, index=False)

    print(f"Separador detectado: {separador!r}")
    print(f"Arquivo normalizado salvo em: {caminho_saida}")


if __name__ == "__main__":
    main()
