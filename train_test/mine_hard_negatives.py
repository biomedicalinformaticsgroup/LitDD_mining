from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives
from datasets import Dataset, load_from_disk
import pandas as pd

ds_cross = load_from_disk('ds_cross_train')
g2p = pd.read_csv('path_to_ddg2p_csv')
g2p["g2p_lgmde"] = g2p.apply(
    lambda row: " - ".join(str(x) for x in [
        row["g2p id"], 
        row["gene symbol"], 
        row["gene mim"],
        row["hgnc id"],
        row["previous gene symbols"],
        row["disease name"],
        row["disease mim"],
        row["disease MONDO"],
        row["allelic requirement"],
        row["cross cutting modifier"],
        row["confidence"],
        row["inferred variant consequence"],
        row["variant types"],
        row["molecular mechanism"],
        row["molecular mechanism categorisation"]
    ]),
    axis=1
)

embedding_model = SentenceTransformer("abhinand/MedEmbed-large-v0.1")
hard_negatives_dataset = mine_hard_negatives(dataset=ds_cross,
                                             model=embedding_model,
                                             anchor_column_name = "tiab",
                                             positive_column_name = "g2p_lgmde",
                                             corpus = list(g2p["g2p_lgmde"]), # additional candidate negatives list
                                             range_min=5,
                                             range_max=50,
                                             max_score=0.95,
                                             relative_margin=0.01,# 0.05 means that the negative is at most 95% as similar to the anchor as the positive 
                                             num_negatives=5, # 10 or less is recommended 
                                             sampling_strategy="top", # "top" means that we sample the top candidates as negatives 
                                             batch_size=128, 
                                             output_format="labeled-pair", 
                                             use_faiss=False)

hard_negatives_dataset.save_to_disk('hard_negatives_dataset')
print('saved hard negatives dataset')