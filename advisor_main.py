import requests
import pandas as pd
import numpy as np

category = ["Childrens Fund",
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
            "Equity: Sectoral-Infrastructure",
            "Equity: Sectoral-Pharma and Healthcare",
            "Equity: Sectoral-Technology",
            "Equity: Small Cap",
            "Equity: Thematic-Consumption",
            "Equity: Thematic-Energy",
            "Equity: Thematic-International",
            "Equity: Thematic-MNC",
            "Equity: Thematic-Others",
            "Equity: Thematic-PSU",
            "Equity: Value",
            "ETFs",
            "Fund of Funds-Domestic-Debt",
            "Fund of Funds-Domestic-Equity",
            "Fund of Funds-Domestic-Gold",
            "Fund of Funds-Domestic-Hybrid",
            "Fund of Funds-Overseas",
            "Hybrid: Aggressive",
            "Hybrid: Arbitrage",
            "Hybrid: Conservative",
            "Hybrid: Dynamic Asset Allocation",
            "Hybrid: Equity Savings",
            "Hybrid: Multi Asset Allocation",
            "Index Fund",
            "Retirement Fund"]


def fetch_data():
    url = "https://www.advisorkhoj.com/mutual-funds-research/" \
          "top-performing-mutual-funds"
    # "?category=Equity:%20Multi%20Cap&period=1y&type=&mode=Growth&option=Regular"

    captured_data = []
    for ele in category:
        payload = {
            "category": ele,
            "period": "1y",
            "type": "",
            "mode": "Growth",
            "option": "Direct"
        }
        resp = requests.get(url, params=payload)
        df_list = pd.read_html(resp.text)
        if df_list:
            df = df_list[0]
            df["", "Category"] = ele
            captured_data.append(df)

    combined = pd.DataFrame()
    for each_df in captured_data:
        combined = combined.append(each_df)
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
        df_final = df_final.append(df_temp)
        df_final = df_final.append(df_temp2)

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
        df_final2 = df_final2.append(df_temp)
        df_final2 = df_final2.append(df_temp2)

    df_final3 = pd.DataFrame()
    df_temp = df_final2.loc[~pd.isna(df_final2['AUM (Crore)'])].copy()
    df_temp2 = df_final2.loc[pd.isna(df_final2['AUM (Crore)'])].copy()
    for col in rank_columns:
        df_temp[f"{col}_rnkAll"] = df_temp[col].rank(ascending=False, method='max')
    for col in rank_columns:
        df_temp[f"{col}_normAll"] = (df_temp[col] - df_temp[col].mean()) / df_temp[col].std()
    df_temp_values = df_temp.loc[:, [f"{col}_normAll" for col in rank_columns[:-1]]]
    df_temp['all_pval'] = df_temp_values.mean(axis=1) / df_temp_values.std(axis=1)
    df_final3 = df_final3.append(df_temp)
    df_final3 = df_final3.append(df_temp2)
    return df_final3


def main():
    # excel_file = "raw_all.xlsx"
    df_all = fetch_data()
    # with pd.ExcelWriter(excel_file,
    #                     engine='xlsxwriter',
    #                     options={'strings_to_numbers': True}) as writer:
    #     df_all.to_excel(writer)

    # input("Correct columns and save excel file")
    # df_all = pd.read_excel(excel_file)
    df_all.columns = ['Scheme Name', "AUM (Crore)", "Expense Ratio (%)", "1 Year", "1 Year_rnk",
                      "3 Years", "3 Years_rnk", "5 Years", "5 Years_rnk", "8 Years", "8 Years_rnk",
                      "Since Launch Rtn. (%)", "Category"]
    df = df_all.drop(["1 Year_rnk", "3 Years_rnk", "5 Years_rnk", "8 Years_rnk"], axis=1)
    df_ranked = rank(df)
    with pd.ExcelWriter("ranked_norm_fullout.xlsx",
                        engine='xlsxwriter',
                        options={'strings_to_numbers': True}) as writer:
        df_ranked.to_excel(writer)


if __name__ == "__main__":
    main()
