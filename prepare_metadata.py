import pandas as pd


def create_metadata():
    df = pd.read_csv("/network/scratch/y/yuyan.chen/anuraset/metadata.csv")

    df_INCT20955 = df[df["site"] == "INCT20955"]  # training
    df_INCT17 = df[df["site"] == "INCT17"]  # testing

    print(df_INCT20955[df_INCT20955["subset"] == "train"])
    print(df_INCT17[df_INCT17["subset"] == "train"])

    print(df_INCT20955[df_INCT20955["subset"] == "test"])
    print(df_INCT17[df_INCT17["subset"] == "test"])

    # # get id species
    # df_INCT20955 = df_INCT20955.loc[
    #     :, (df_INCT20955 != 0).any(axis=0)
    # ]  # remove all all-zero columns

    # A = set(df_INCT20955.iloc[:, 8:].columns)

    # df_INCT17 = df_INCT17.loc[:, (df_INCT17 != 0).any(axis=0)]

    # B = set(df_INCT17.iloc[:, 8:].columns)

    # print(A & B)

    # print(len(B))

    # df_INCT17.to_csv(
    #     "/network/scratch/y/yuyan.chen/anuraset/INCT17_metadata.csv", index=False
    # )
    # df_INCT20955.to_csv(
    #     "/network/scratch/y/yuyan.chen/anuraset/INCT20955_metadata.csv", index=False
    # )


def get_label():
    df = pd.read_csv("/network/scratch/y/yuyan.chen/anuraset/metadata.csv")
    df_INCT20955 = df[df["site"] == "INCT20955"]  # training
    df_INCT17 = df[df["site"] == "INCT17"]  # testing

    df_INCT20955 = df_INCT20955.loc[:, (df_INCT20955 != 0).any(axis=0)]

    ood_species = list(set(df_INCT17.columns) - set(df_INCT20955.columns))
    id_species = df_INCT20955.columns[8:]

    df_id = df_INCT17[id_species]

    df_ood = df_INCT17[ood_species]

    df_INCT17["has_ood"] = df_ood.any(axis=1)
    df_INCT17["has_id"] = df_id.any(axis=1)

    print(df_INCT17[ood_species].iloc[0])

    print(len(df_INCT17))
    print(len(df_INCT17[df_INCT17["has_ood"] & df_INCT17["has_id"]]))
    print(len(df_INCT17[df_INCT17["has_ood"] & ~df_INCT17["has_id"]]))
    print(len(df_INCT17[~df_INCT17["has_ood"] & df_INCT17["has_id"]]))
    print(len(df_INCT17[~df_INCT17["has_ood"] & ~df_INCT17["has_id"]]))

    # df_INCT17.to_csv(
    #     "/network/scratch/y/yuyan.chen/anuraset/INCT17_metadata_w_ol.csv", index=False
    # )


def setting2_metadata():
    df = pd.read_csv("/network/scratch/y/yuyan.chen/anuraset/metadata.csv")
    save_dir = "/network/scratch/y/yuyan.chen/anuraset/setting2"
    df_INCT20955 = df[df["site"] == "INCT20955"]
    df_INCT20955 = df_INCT20955.loc[:, (df_INCT20955 != 0).any(axis=0)]
    species = set(df_INCT20955.iloc[:, 8:].columns)
    OOD_species = set(["DENMIN", "PHYCUV", "LEPLAT", "SCIPER", "BOAPRA", "RHIICT"])
    ID_species = species - OOD_species

    print(ID_species)

    df_ood = df_INCT20955[list(OOD_species)]
    df_id = df_INCT20955[list(ID_species)]

    df_INCT20955["has_ood"] = df_ood.any(axis=1)
    df_INCT20955["has_id"] = df_id.any(axis=1)

    df_both = df_INCT20955[df_INCT20955["has_ood"] & df_INCT20955["has_id"]]
    df_id_only = df_INCT20955[~df_INCT20955["has_ood"]]
    df_ood_only = df_INCT20955[df_INCT20955["has_ood"] & ~df_INCT20955["has_id"]]

    assert len(df_both) + len(df_id_only) + len(df_ood_only) == len(df_INCT20955)

    # train/ val/ test split
    id_train = df_id_only.sample(frac=0.7, replace=False, random_state=42)
    temp = df_id_only.drop(index=id_train.index)
    id_val = temp.sample(frac=0.5, replace=False, random_state=42)
    id_test = temp.drop(index=id_val.index)

    assert len(id_train) + len(id_val) + len(id_test) == len(df_id_only)

    id_train = id_train.assign(subset="train")
    id_val = id_val.assign(subset="val")
    id_test = id_test.assign(subset="test")

    id_train.to_csv(f"{save_dir}/id_train.csv", index=False)
    id_val.to_csv(f"{save_dir}/id_val.csv", index=False)
    id_test.to_csv(f"{save_dir}/id_test.csv", index=False)

    ood_val = df_ood_only.sample(frac=0.5, replace=False, random_state=42)
    ood_test = df_ood_only.drop(index=ood_val.index)

    assert len(ood_val) + len(ood_test) == len(df_ood_only)

    ood_val = ood_val.assign(subset="val")
    ood_test = ood_test.assign(subset="val")
    ood_val.to_csv(f"{save_dir}/ood_val.csv", index=False)
    ood_test.to_csv(f"{save_dir}/ood_test.csv", index=False)

    both_val = df_both.sample(frac=0.5, replace=False, random_state=42)
    both_test = df_both.drop(index=both_val.index)

    assert len(both_val) + len(both_test) == len(df_both)

    both_val = both_val.assign(subset="val")
    both_test = both_test.assign(subset="val")
    both_val.to_csv(f"{save_dir}/both_val.csv", index=False)
    both_test.to_csv(f"{save_dir}/both_test.csv", index=False)


if __name__ == "__main__":
    setting2_metadata()
