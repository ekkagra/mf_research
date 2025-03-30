from __future__ import annotations

import requests
import pandas as pd
import numpy as np
from datetime import datetime as dt
import argparse
from typing import Callable

category = [
    "Childrens Fund",
    "Debt: Banking and PSU",
    "Debt: Corporate Bond",
    "Debt: Credit Risk",
    "Debt: Dynamic Bond",
    "Debt: Floater",
    "Debt: Gilt",
    "Debt: Gilt Fund with 10 year constant duration",
    "Debt: Liquid",
    "Debt: Long Duration",
    "Debt: Low Duration",
    "Debt: Medium Duration",
    "Debt: Medium to Long Duration",
    "Debt: Money Market",
    "Debt: Overnight",
    "Debt: Short Duration",
    "Debt: Ultra Short Duration",
    "Equity: Contra",
    "Equity: Dividend Yield",
    "Equity: ELSS",
    "Equity: Flexi Cap",
    "Equity: Focused",
    "Equity: Large and Mid Cap",
    "Equity: Large Cap",
    "Equity: Mid Cap",
    "Equity: Multi Cap",
    "Equity: Sectoral-Banking and Financial Services",
    "Equity: Sectoral-FMCG",
    "Equity: Sectoral-Infrastructure",
    "Equity: Sectoral-Pharma and Healthcare",
    "Equity: Sectoral-Technology",
    "Equity: Small Cap",
    "Equity: Thematic-Consumption",
    "Equity: Thematic-Energy",
    "Equity: Thematic-ESG",
    "Equity: Thematic-International",
    "Equity: Thematic-Manufacturing",
    "Equity: Thematic-MNC",
    "Equity: Thematic-Multi-Sector",
    "Equity: Thematic-Others",
    "Equity: Thematic-PSU",
    "Equity: Thematic-Quantitative",
    "Equity: Thematic-Transportation",
    "Equity: Value",
    "ETFs",
    "Fund of Funds-Domestic-Debt",
    "Fund of Funds-Domestic-Equity",
    "Fund of Funds-Domestic-Gold",
    "Fund of Funds-Domestic-Gold and Silver",
    "Fund of Funds-Domestic-Hybrid",
    "Fund of Funds-Domestic-Silver",
    "Fund of Funds-Overseas",
    "Hybrid: Aggressive",
    "Hybrid: Arbitrage",
    "Hybrid: Balanced",
    "Hybrid: Conservative",
    "Hybrid: Dynamic Asset Allocation",
    "Hybrid: Equity Savings",
    "Hybrid: Multi Asset Allocation",
    "Index Fund",
    "Retirement Fund",
]


COLUMN_NAMES = ['Scheme Name',
                "Launch date",
                "AUM (Crore)",
                "Expense Ratio (%)",
                "1 Year",
                "1 Year_rnk",
                "3 Years",
                "3 Years_rnk",
                "5 Years",
                "5 Years_rnk",
                "8 Years",
                "8 Years_rnk",
                "Since Launch Rtn. (%)", "Category"]
COLUMN_NAMES2 = ['Scheme Name',
                 "Sub Category",
                 "Launch date",
                 "AUM (Crore)",
                 "Expense Ratio (%)",
                 "1 Year",
                 "3 Years",
                 "5 Years",
                 "8 Years",
                 "Since Launch Rtn. (%)",
                 "Category"]

FINAL_COLUMN_NAMES = ['Scheme Name',
                      "Sub Category",
                      "Launch date",
                      "AUM (Crore)",
                      "Expense Ratio (%)",
                      "1 Year",
                      "1 Year_rnk",
                      "3 Years",
                      "3 Years_rnk",
                      "5 Years",
                      "5 Years_rnk",
                      "8 Years",
                      "8 Years_rnk",
                      "Since Launch Rtn. (%)", "Category"]


def fetch_data(base_url: str, payload_gen: Callable) -> list[tuple[str, pd.DataFrame]]:
    captured_data: list[tuple[str, pd.DataFrame]] = []
    for cat in category:

        payload: dict = payload_gen(cat)
        resp = requests.get(base_url, params=payload)
        df_list = pd.read_html(resp.text)
        for df in df_list:
            captured_data.append((cat, df))
    return captured_data


def sanitize_trailing_data(dfs: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    captured_data = []
    for cat, df in dfs:
        if cat not in ["ETFs", "Index Fund"]:
            df["", "Category"] = cat
            df.columns = COLUMN_NAMES
            df["Sub Category"] = cat
            df = df[FINAL_COLUMN_NAMES]
            captured_data.append(df)
        else:
            df["Category"] = cat
            df.columns = COLUMN_NAMES2
            df["1 Year_rnk"] = np.nan
            df["3 Years_rnk"] = np.nan
            df["5 Years_rnk"] = np.nan
            df["8 Years_rnk"] = np.nan
            df = df[FINAL_COLUMN_NAMES]
            captured_data.append(df)

    combined = pd.DataFrame()
    for each_df in captured_data:
        if len(each_df) != 0:
            combined = pd.concat([combined, each_df], ignore_index=True)
    return combined

def sanitize_annualized_data(dfs: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    COLUMNS = [
        'Scheme Name',
        "AMC Name",
        "Lauch Date",
        "AUM (Crore)",
        "TER (%)",
        "2025",
        "2024",
        "2023",
        "2022",
        "2021",
        "Category"
    ]
    FINAL = [
        'Scheme Name',
        "Category",
        "AMC Name",
        "Lauch Date",
        "AUM (Crore)",
        "TER (%)",
        "2025",
        "2024",
        "2023",
        "2022",
        "2021",
    ]

    to_sanitize_cols = [
        "AUM (Crore)",
        "TER (%)",
        "2025",
        "2024",
        "2023",
        "2022",
        "2021",
    ]

    captured_data = []
    for cat, df in dfs:
        print(cat, "records=", len(df))
        df["", "Category"] = cat
        df.dropna(axis=1, thresh=5, inplace=True)
        if len(df) < 2 or len(df.columns) < 5:
            continue
        df.columns = COLUMNS
        df = df[FINAL]

        for col in to_sanitize_cols:
            df[col] = df[col].apply(convert_to_null)

        captured_data.append(df)

    combined = pd.DataFrame()
    for each_df in captured_data:
        if len(each_df) != 0:
            combined = pd.concat([combined, each_df], ignore_index=True)
    return combined


def label_major_category(x):
    if "debt" in str(x).strip().lower():
        return "Debt"
    elif "hybrid" in str(x).strip().lower():
        return "Hybrid"
    else:
        return "Equity"

def convert_to_null(x):
    if str(x).strip() == '-':
        return np.nan
    elif str(x).strip() == "":
        return np.nan
    else:
        return x

def rank(df):
    columns_to_apply = ['AUM (Crore)', "Expense Ratio (%)", '1 Year', '3 Years', '5 Years', '8 Years',
                        'Since Launch Rtn. (%)']
    for col in columns_to_apply:
        df[col] = df[col].apply(convert_to_null)
    df["MajorCategory"] = df["Category"].apply(label_major_category)
    categories = df['Category'].unique()
    rank_columns = ['1 Year', '3 Years', '5 Years', '8 Years', 'Since Launch Rtn. (%)']
    df = df.astype({"AUM (Crore)": float,
                    "Expense Ratio (%)": float,
                    "1 Year": float,
                    "3 Years": float,
                    "5 Years": float,
                    "8 Years": float,
                    "Since Launch Rtn. (%)": float})
    df_original = df.copy()

    df_final = pd.DataFrame()
    for cat in categories:
        df_temp_cat = df.loc[df['Category'] == cat].copy()
        df_temp = df_temp_cat.loc[~pd.isna(df_temp_cat['AUM (Crore)'])].copy()
        df_temp2 = df_temp_cat.loc[pd.isna(df_temp_cat['AUM (Crore)'])].copy()
        for col in rank_columns:
            df_temp[f"{col}_rnkCat"] = df_temp[col].rank(ascending=False, method='max')
        for col in rank_columns:
            df_temp[f"{col}_normCat"] = (df_temp[col] - df_temp[col].mean()) / df_temp[col].std()
        df_temp_values = df_temp.loc[:, [f"{col}_normCat" for col in rank_columns[:-1]]]
        df_temp['Category_pval'] = df_temp_values.mean(axis=1) / df_temp_values.std(axis=1)
        df_temp['Category_pvalNorm'] = (df_temp['Category_pval'] - df_temp['Category_pval'].mean()) / df_temp[
            'Category_pval'].std()
        df_final = pd.concat([df_final, df_temp], ignore_index=True)
        df_final = pd.concat([df_final, df_temp2], ignore_index=True)

    df_final2 = pd.DataFrame()
    for cat in ["Debt", "Equity", "Hybrid"]:
        df_temp_cat = df_final.loc[df['MajorCategory'] == cat].copy()
        df_temp = df_temp_cat.loc[~pd.isna(df_temp_cat['AUM (Crore)'])].copy()
        df_temp2 = df_temp_cat.loc[pd.isna(df_temp_cat['AUM (Crore)'])].copy()
        for col in rank_columns:
            df_temp[f"{col}_rnkMajor"] = df_temp[col].rank(ascending=False, method='max')
        for col in rank_columns:
            df_temp[f"{col}_normMajor"] = (df_temp[col] - df_temp[col].mean()) / df_temp[col].std()
        df_temp_values = df_temp.loc[:, [f"{col}_normMajor" for col in rank_columns[:-1]]]
        df_temp['majCat_pval'] = df_temp_values.mean(axis=1) / df_temp_values.std(axis=1)
        df_temp['majCat_pvalNorm'] = (df_temp['majCat_pval'] - df_temp['majCat_pval'].mean()) / df_temp[
            'majCat_pval'].std()
        df_final2 = pd.concat([df_final2, df_temp], ignore_index=True)
        df_final2 = pd.concat([df_final2, df_temp2], ignore_index=True)

    df_final3 = pd.DataFrame()
    df_temp = df_final2.loc[~pd.isna(df_final2['AUM (Crore)'])].copy()
    df_temp2 = df_final2.loc[pd.isna(df_final2['AUM (Crore)'])].copy()
    for col in rank_columns:
        df_temp[f"{col}_rnkAll"] = df_temp[col].rank(ascending=False, method='max')
    for col in rank_columns:
        df_temp[f"{col}_normAll"] = (df_temp[col] - df_temp[col].mean()) / df_temp[col].std()
    df_temp_values = df_temp.loc[:, [f"{col}_normAll" for col in rank_columns[:-1]]]
    df_temp['all_pval'] = df_temp_values.mean(axis=1) / df_temp_values.std(axis=1)
    df_temp['all_pvalNorm'] = (df_temp['all_pval'] - df_temp['all_pval'].mean()) / df_temp['all_pval'].std()
    df_final3 = pd.concat([df_final3, df_temp], ignore_index=True)
    df_final3 = pd.concat([df_final3, df_temp2], ignore_index=True)
    return df_final3

def process_trailing():
    print("Fetching trailing returns")

    base_url = "https://www.advisorkhoj.com/mutual-funds-research/top-performing-mutual-funds"
    # "?category=Equity:%20Multi%20Cap&period=1y&type=&mode=Growth&option=Regular"

    def payload_gen(cat: str) -> dict:
        payload = {
            "category": cat,
            "period": "1y",
            "type": "",
            "mode": "Growth",
            "option": "Direct"
        }
        if cat in ["ETFs", "Index Fund"]:
            payload["subcategory"] = "All"
        return payload

    dfs = fetch_data(base_url, payload_gen)
    df_all = sanitize_trailing_data(dfs)

    df_all.columns = FINAL_COLUMN_NAMES
    df = df_all.drop(["1 Year_rnk", "3 Years_rnk", "5 Years_rnk", "8 Years_rnk", "Launch date", "Sub Category"], axis=1)
    df_ranked = rank(df)
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    with pd.ExcelWriter(f"trail_ranked_norm_fullout_{timestamp}.xlsx",
                        engine='xlsxwriter',
                        engine_kwargs={'options':{'strings_to_numbers': True}}) as writer:
        df_ranked.to_excel(writer)

def process_annualized():
    print("Fetching annualized returns")
    # https://www.advisorkhoj.com/mutual-funds-research/mutual-fund-annual-returns?category=Index-Fund&scheme_plan_type=Direct

    def payload_gen(cat: str) -> dict:
        payload = {
            "category": cat,
            "scheme_plan_type": "Direct",
        }
        return payload

    dfs = fetch_data("https://www.advisorkhoj.com/mutual-funds-research/mutual-fund-annual-returns", payload_gen)
    df_all = sanitize_annualized_data(dfs)

    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    with pd.ExcelWriter(f"annual_ranked_norm_fullout_{timestamp}.xlsx",
                        engine='xlsxwriter',
                        engine_kwargs={'options':{'strings_to_numbers': True}}) as writer:
        df_all.to_excel(writer)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trail", action="store_true")
    parser.add_argument("--annual", action="store_true")
    args = parser.parse_args()

    if not args.trail and not args.annual:
        parser.print_usage()
        return

    if args.trail:
        process_trailing()

    if args.annual:
        process_annualized()

if __name__ == "__main__":
    main()
