import pandas as pd

df = pd.read_csv("/network/scratch/y/yuyan.chen/anuraset/metadata.csv")


df_INCT20955 = df[df["site"] == "INCT20955"]  # training
df_INCT17 = df[df["site"] == "INCT17"]  # testing

df_INCT20955 = df_INCT20955.loc[
    :, (df_INCT20955 != 0).any(axis=0)
]  # remove all all-zero columns

df_INCT17.to_csv(
    "/network/scratch/y/yuyan.chen/anuraset/INCT17_metadata.csv", index=False
)
df_INCT20955.to_csv(
    "/network/scratch/y/yuyan.chen/anuraset/INCT20955_metadata.csv", index=False
)
