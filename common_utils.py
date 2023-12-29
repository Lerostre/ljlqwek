import pandas as pd
import numpy as np

asp_mapper = dict(zip(['Whole', 'Interior', 'Service', 'Food', 'Price', np.nan], range(6)))
sent_mapper = dict(zip(["positive", "negative", "both", "neutral", "absence", np.nan], range(6)))

# asp_mapper = dict(zip(['Whole', 'Interior', 'Service', 'Food', 'Price', np.nan], range(6)))
# sent_mapper = {"positive": 1, "negative": 0, "both": 2, "neutral": 3, np.nan: 4}

def correct_aspects(aspects):
    res = pd.DataFrame(columns=["text_id", "aspect", "token", "begin", "end", "sentiment"])

    for idx, row in aspects.iterrows():
        row = aspects.iloc[idx]
        tokens = row["token"].split()
        char_id = row["begin"]

        for token_id, token in enumerate(tokens):
            # prefix = {len(tokens)-1: "E", 0: "B"}
            res.loc[res.shape[0]] = {
                "text_id": row["text_id"],
                # "aspect": f"{prefix.get(token_id, 'I')}-{row['aspect']}",
                "aspect": row['aspect'],
                "token": token,
                "sentiment": row["sentiment"],
                "begin": char_id,
                "end": char_id + len(token)
            }
            char_id += len(token) + 1
            
    return res