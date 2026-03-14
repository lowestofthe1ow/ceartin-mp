import pandas as pd


def wikipron_tl_df(wikipron_path):
    # Read from file and drop all duplicates
    raw_df = pd.read_csv(wikipron_path, sep="\t", header=None, names=["word", "pron"])
    raw_df = raw_df.drop_duplicates(keep="first").reset_index(drop=True)

    # Identify all indices where spelling is identical
    homograph_mask = raw_df["word"].duplicated(keep=False)

    homographs = raw_df[homograph_mask].reset_index(drop=True)
    non_homographs = raw_df[~homograph_mask].reset_index(drop=True)

    # Create dicts from the DataFrames for more efficient lookup later
    homographs = homographs.groupby("word")["pron"].apply(list).to_dict()
    non_homographs = non_homographs.set_index("word")["pron"].to_dict()

    """
    print("=" * 40)
    print("Processed WikiPron Tagalog dataset.")
    print("-" * 40)
    print(f"Homographs (unique): {len(homographs)}")
    print(f"Non-homographs: {len(non_homographs)}")
    print("=" * 40)
    """

    return homographs, non_homographs


# If run as a script, show information about the WikiPron data...
if __name__ == "__main__":
    file_path = "data/wikipron/wikipron_tl.tsv"
    homo_dict, non_homo_dict = wikipron_tl_df(file_path)

    all_pronunciations = list(non_homo_dict.values())

    for prons in homo_dict.values():
        all_pronunciations.extend(prons)

    # Extract unique characters
    phoneme_set = set()
    for pron in all_pronunciations:
        # Update the set with every character in the string, ignoring spaces
        phoneme_set.update(char for char in pron if char != " ")

    phoneme_inventory = sorted(list(phoneme_set))

    print("=" * 40)
    print("Phoneme Character Inventory:")
    print("=" * 40)
    print(f"Total unique characters: {len(phoneme_inventory)}")
    print(phoneme_inventory)
