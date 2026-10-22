GI50 = "/Users/andreabenassi/OMIRA-q/data/input/GI50.csv"
LEGACY = "/Users/andreabenassi/OMIRA-q/data/input/nci60_full_dataset_OLD.csv"

import pandas as pd
gi = pd.read_csv(GI50, usecols=["CELL_NAME", "PANEL_CODE"]).drop_duplicates()
PREFIX = {"BRE": "BR", "CNS": "CNS", "COL": "CO", "LEU": "LE", "MEL": "ME",
          "LNS": "LC", "OVA": "OV", "PRO": "PR", "REN": "RE"}
gi["PREFIX"] = gi["PANEL_CODE"].map(PREFIX)
print(gi[gi["PREFIX"].isna()]["PANEL_CODE"].value_counts())