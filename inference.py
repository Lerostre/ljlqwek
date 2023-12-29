from collections import defaultdict, Counter
from string import punctuation
from tqdm.auto import tqdm
import torch
import pandas as pd
import numpy as np

from datasets import Dataset
from common_utils import correct_aspects, asp_mapper, sent_mapper
from bert import tokenize_and_align_labels
from torch.utils.data import DataLoader
from functools import reduce
from bert import evaluate
from pathlib import Path
import wandb
from pytorch_lightning.loggers import WandbLogger

def mention_sentiment(matches):
    matched_sentiment = 0.
    for pair in matches:
        *_, gold_s = pair[0]
        *_, pred_s = pair[1]
        if gold_s == pred_s:
            matched_sentiment += 1
    return matched_sentiment / len(matches)


def get_full_sentiment(frame, aspect_col, sentiment_col, max_len=5):
    
    # функция изначально кривая, в ней лишние аргументы
    # будем считать, что на вход она принимает датафрейм с аспектами
    CATEGORIES = ['Whole', 'Interior', 'Service', 'Food', 'Price']
    
    asp_counter = defaultdict(Counter)
    aspect_sentiments = []
    frame = frame[["text_id", aspect_col, sentiment_col]][frame[aspect_col].notna()]
    frame = frame.reset_index(drop=True)
    
    for text_id in frame["text_id"].unique():
        for row in frame[frame.text_id == text_id].index:
            category, sentiment = frame.loc[row, [aspect_col, sentiment_col]]
            asp_counter[category][sentiment] += 1
        for c in CATEGORIES:
            if not asp_counter[c]:
                s = 'absence'
            elif len(asp_counter[c]) == 1:
                s = asp_counter[c].most_common(1)[0][0]
            else:
                s = 'both'
            yield text_id, c, s


def overall_sentiment_accuracy(gold_test_cats, pred_test_cats):
    with open(gold_test_cats, "r") as gc, open(pred_test_cats, "r") as pc:
        # она игнорирует, что там за аспекты, опять фиксить
        gold_labels = set(gc.readlines())
        pred_labels = set(pc.readlines())
        return len(gold_labels & pred_labels) / len(gold_labels)


def reference_check(
    gold_test_path, pred_test_path,
    gold_test_cats, pred_test_cats, model=None
):

    gold_aspect_cats = {}
    with open(gold_test_path) as fg:
        for line in fg:
            line = line.rstrip('\r\n').split('\t')
            if line[0] not in gold_aspect_cats:
                gold_aspect_cats[line[0]] = {"starts":[], "ends":[], "cats":[], "sents":[]}
            gold_aspect_cats[line[0]]["starts"].append(int(line[3]))
            gold_aspect_cats[line[0]]["ends"].append(int(line[4]))
            gold_aspect_cats[line[0]]["cats"].append(line[1])
            gold_aspect_cats[line[0]]["sents"].append(line[5])

    full_match, partial_match, full_cat_match, partial_cat_match = 0, 0, 0, 0
    total = 0
    fully_matched_pairs = []
    partially_matched_pairs = []
    with open(pred_test_path) as fp:
        for line in fp:    
            total += 1
            line = line.rstrip('\r\n').split('\t')
            start, end = int(line[3]), int(line[4])
            category = line[1]
            doc_gold_aspect_cats = gold_aspect_cats[line[0]]

            if start in doc_gold_aspect_cats["starts"]:
                i = doc_gold_aspect_cats["starts"].index(start)
                if doc_gold_aspect_cats["ends"][i] == end:
                    full_match += 1
                    if doc_gold_aspect_cats["cats"][i] == category:
                        full_cat_match += 1
                    else:
                        partial_cat_match += 1
                    fully_matched_pairs.append((
                        [doc_gold_aspect_cats["starts"][i], 
                         doc_gold_aspect_cats["ends"][i], 
                         doc_gold_aspect_cats["cats"][i],
                         doc_gold_aspect_cats["sents"][i]],
                        line
                    ))
                    continue

            for s_pos in doc_gold_aspect_cats["starts"]:
                if start <= s_pos:
                    i = doc_gold_aspect_cats["starts"].index(s_pos)
                    if doc_gold_aspect_cats["ends"][i] == end:
                        partial_match += 1
                        partially_matched_pairs.append((
                            [doc_gold_aspect_cats["starts"][i], 
                             doc_gold_aspect_cats["ends"][i], 
                             doc_gold_aspect_cats["cats"][i],
                             doc_gold_aspect_cats["sents"][i]],
                            line
                        ))
                        if doc_gold_aspect_cats["cats"][i] == category:
                            partial_cat_match += 1
                        continue
                    matched = False
                    for e_pos in doc_gold_aspect_cats["ends"][i:]:
                        if s_pos <= end <= e_pos:
                            partial_match += 1
                            partially_matched_pairs.append(
                                (
                                    [
                                        doc_gold_aspect_cats["starts"][i], 
                                        doc_gold_aspect_cats["ends"][i], 
                                        doc_gold_aspect_cats["cats"][i],
                                        doc_gold_aspect_cats["sents"][i]
                                    ],
                                    line
                                )
                            )
                            if doc_gold_aspect_cats["cats"][i] == category:
                                partial_cat_match += 1
                            matched = True
                            break
                    if matched:
                        break
                if start > s_pos:
                    i = doc_gold_aspect_cats["starts"].index(s_pos)
                    if start < doc_gold_aspect_cats["ends"][i] <= end:
                        partial_match += 1
                        partially_matched_pairs.append(
                            (
                                [
                                    doc_gold_aspect_cats["starts"][i], 
                                    doc_gold_aspect_cats["ends"][i], 
                                    doc_gold_aspect_cats["cats"][i],
                                    doc_gold_aspect_cats["sents"][i]
                                ],
                                line
                            )
                        )
                        if doc_gold_aspect_cats["cats"][i] == category:
                            partial_cat_match += 1
                        break

    gold_size = sum([len(gold_aspect_cats[x]["cats"]) for x in gold_aspect_cats])
    res = map(lambda x: x.split(": "), [
        f"Full match precision: {full_match / total}",
        f"Full match recall: {full_match / gold_size}",
        f"Partial match ratio in pred: {(full_match + partial_match)  / total}",
        f"Full category accuracy: {full_cat_match / total}",
        f"Partial category accuracy: {(full_cat_match + partial_cat_match) / total}",
        f"Patial sentiment accuracy: {mention_sentiment(partially_matched_pairs)}",
        f"Full sentiment accuracy: {mention_sentiment(fully_matched_pairs)}",
        f"Overall sentiment accuracy: {overall_sentiment_accuracy(gold_test_cats, pred_test_cats)}",
    ])
    res = pd.DataFrame([dict(res)]).T.rename(columns={0: model})
    return res
    

def custom_accuracy(pred, true, how="pred"):
    accuracy = []
    method = {
        "true": lambda x, y: pd.isna(x),
        "pred": lambda x, y: pd.isna(y),
        "common": lambda x, y: pd.isna(y) or pd.isna(x),
        "excluding": lambda x, y: pd.isna(y) and pd.isna(x),
    }
    compare = method[how]
    for pred_id, true_id in zip(pred, true):
        if not compare(pred_id, true_id):
            accuracy.append(pred_id == true_id)
    return torch.tensor(accuracy).sum() / len(accuracy)


def pack(frame, aspect_col, sentiment_col, output_file):
    
    meta_cols = ["text_id", "token", "begin", "end"]
    packed_aspects = pd.DataFrame(columns=meta_cols)
    frame = frame[meta_cols+[aspect_col, sentiment_col]].dropna()
    frame = frame.reset_index(drop=True)
    n_of_rows = frame.shape[0]
    
    with open(output_file, "w") as f:
        text_id, token, begin, end = 0, 0, 0, 0
        for row_idx in tqdm(range(n_of_rows)):
            if not begin or frame.loc[row_idx, "text_id"] != text_id:
                text_id, token, begin, end = frame.loc[row_idx, meta_cols]
            if row_idx != n_of_rows-1:
                candidate = frame.loc[row_idx+1, "begin"] - 1
                aspect, sentiment = frame.loc[row_idx, [aspect_col, sentiment_col]]
                if pd.isna(aspect) or pd.isna(sentiment):
                    continue
                elif end == candidate:
                    end = frame.loc[row_idx+1, "end"]
                    token += " "+frame.loc[row_idx+1, "token"]
                else:
                    res = {
                        "text_id": frame.loc[row_idx, "text_id"],
                        "aspect": aspect, "token": token.strip(".?!,"),
                        "begin": begin, "end": end, "sentiment": sentiment
                    }
                    print(*res.values(), sep="\t", file=f)
                    begin = 0   
                    

def inference(texts: str, prefix="test"):

    texts = pd.read_csv(texts, delimiter="\t", names=["text_id", "token"])
    texts["text"] = texts["token"]
    texts["token"] = texts["token"].map(lambda x: x.split())
    texts = texts.explode("token").reset_index(drop=True)

    text_id = 0
    for idx, row in texts.iterrows():

        if row["text_id"] != text_id:
            text_id = row["text_id"]
            char_id = 0
        texts.loc[idx, "begin"] = char_id
        token = row["token"]
        texts.loc[idx, "end"] = char_id + len(token.strip(punctuation))

        char_id += len(token) + 1

    texts[["begin", "end"]] = texts[["begin", "end"]].astype(int)
    texts["token"] = texts.token.apply(lambda x: x.strip(punctuation))

    dataset = texts.groupby("text_id").agg(lambda x: [y for y in x]).reset_index()
    dataset = Dataset.from_pandas(dataset)
    dataset = dataset.map(tokenize_and_align_labels, batched=True)
    dataset.set_format("pt")
    
    meta_cols = ["begin", "end"]
    meta = pd.DataFrame(columns=meta_cols)
    all_tokens = [y for x in dataset["token"] for y in x if y != "-100"]
    meta["token"] = all_tokens
    for col in meta_cols:
        meta[col] = dataset[col][dataset[col] != -100].detach().numpy()

    cur_idx = -1
    ids = np.unique(texts.text_id)
    for idx, row in meta.iterrows():
        if row["begin"] == 0:
            cur_idx += 1
        meta.loc[idx, "text_id"] = ids[cur_idx]
    
    dataset = dataset.remove_columns(["token", "end", "text_id", "text"])
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    wandb_logger = WandbLogger(
        project="assez", name="cat_bert"
    )
    artifact = wandb_logger.use_artifact("lerostre/assez/model-z10m03iu:best")
    artifact_dir = artifact.download()
    bert_res, clf_bert = evaluate("artifacts/model-z10m03iu:v49/model.ckpt", loader)
    
    meta = meta[["text_id", "token", "begin", "end"]]
    columns = ["text_id", "aspect_pred", "token", "begin", "end", "sentiment_pred"]

    bert_pretty_res = pd.concat([meta, bert_res], axis=1)[columns]
    bert_pretty_res = bert_pretty_res.dropna().reset_index(drop=True)
    bert_pretty_res[["text_id", "begin", "end"]] = bert_pretty_res[["text_id", "begin", "end"]].astype(int)
    
    with open(f'{prefix}_bert_aspects.txt', 'w') as f:
        for idx, l in bert_pretty_res.iterrows():
            print(*l.values, sep="\t", file=f)
        
    with open(f'{prefix}_bert_packed_cats.txt', 'w') as f:
        for idx, row in enumerate(clf_bert["clf_pred"].values):
            for cat, sent in zip(
                ['Food', 'Interior', 'Price', 'Whole', 'Service'], row
            ):
                print(ids[idx], cat, sent, sep="\t", file=f)

    pack(
        bert_pretty_res, "aspect_pred", "sentiment_pred",
        "test_bert_packed_aspects.txt"
    )
    print("Done")