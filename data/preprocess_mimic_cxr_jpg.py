import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Literal, Dict
from nltk.tokenize import RegexpTokenizer
import pickle
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from data.mgca_constants import *
from data.mgca_utils import extract_mimic_text
from tqdm import tqdm
import re


SplitT = Literal["train", "validate", "val", "test"]

@dataclass
class MIMICCXRConfig:
    root: str                                   # root directory that contains the "files/" tree
    metadata_csv: str                           # path to mimic-cxr-2.0.0-metadata.csv.gz
    split_csv: str                              # path to mimic-cxr-2.0.0-split.csv.gz
    image_filenames_txt: str    
    caption_max_len: int                # path to IMAGE_FILENAMES
    label_col: Optional[str] = None             # if None -> multi-label CheXpert; else single-label categorical
    chexpert_csv: Optional[str] = None          # path to mimic-cxr-2.0.0-chexpert.csv.gz for multi-label
    transform: Optional[Callable] = None
    target_transform: Optional[Callable] = None
    drop_missing_labels: bool = True
    verify_data_path: bool = False
    mask_uncertain_labels: bool = False
    override_master_csv: bool = False

class MIMICCXRDataloader(Dataset):
    """
    If cfg.label_col is None and cfg.chexpert_csv is provided -> returns a multi-label torch.FloatTensor
    of the 14 CheXpert labels (values in {1.0, 0.0, -1.0}). Missing values are set to 0.0 by default.

    If cfg.label_col is not None -> returns a single integer class index (torch.long).
    """
    CHEXPERT_LABELS = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
        "Fracture", "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Pneumonia",
        "Pneumothorax", "Pleural Other", "Support Devices", "No Finding"
    ]

    def __init__(self, cfg: MIMICCXRConfig, tokenizer, split: SplitT = "train"):

        self.cfg = cfg
        self.split = "validate" if split == "val" else split
        assert self.split in {"train", "validate", "test"}, f"Invalid split: {split}"

        # Load metadata and split
        metadata = pd.read_csv(cfg.metadata_csv, compression="infer")
        split_df = pd.read_csv(cfg.split_csv, compression="infer")
        split_df = split_df[split_df["split"] == self.split]

        # Map dicom_id to relative paths
        id2relpath = self._load_image_paths(cfg.image_filenames_txt)
        split_df = split_df[split_df["dicom_id"].isin(id2relpath)]

        # Merge metadata
        df = split_df.merge(metadata, on="dicom_id", how="left")
        df["path"] = df["dicom_id"].map(lambda x: os.path.join(cfg.root, id2relpath[x]))
        df = df.drop(columns=["study_id_y", "subject_id_y"])
        df = df.rename(columns={"study_id_x": "study_id", "subject_id_x": "subject_id"})
        
        if not cfg.chexpert_csv:
            raise ValueError("For multi-label mode, you must provide chexpert_csv.")
        chexpert_df = pd.read_csv(cfg.chexpert_csv, compression="infer")
        df = df.merge(chexpert_df, on=["subject_id", "study_id"], how="left")

        # Fill missing labels with 0.0 (not mentioned)
        label_cols = self.CHEXPERT_LABELS
        missing_cols = [c for c in label_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"CheXpert CSV missing expected columns: {missing_cols}")
        self.classes = label_cols

        # replace -1.0 as 0
        df[label_cols] = df[label_cols].fillna(0.0).replace(-1.0, 0.0).astype("float32") if not self.cfg.mask_uncertain_labels else df[label_cols].fillna(0.0).astype("float32")
        
        # Cache the targets as a NumPy array for speed
        self.label_col_names = label_cols

        self.metadata_df = df.reset_index(drop=True)
        self.transform = cfg.transform
        self.target_transform = cfg.target_transform

        self.tokenizer = tokenizer

        # NOTE: the following are the processes for the MGCA Mmodel
        # TODO: should be unique for each method.
        self.master_df = self.preprocess_mimic_csv_files()
        self.filenames, self.path2sent = self.load_text_data('valid' if self.split == 'validate' else self.split)

        # get the metadatadf as dictionary
        self.metadata_dict = {
            row['path']: row for _, row in df.iterrows()
        }
        # TODO: check why there are few files in self.filenames compared to self.metadata_dict
        assert set(self.filenames).issubset(set(self.metadata_dict.keys()))

    def _load_image_paths(self, filelist_path: str) -> Dict[str, str]:
        id2path = {}
        exists = 0
        with open(filelist_path, "r") as f:
            for line in f:
                rel_path = line.strip()

                if not rel_path:
                    continue

                sub_dir = rel_path.removeprefix("files/")
                if not self.cfg.verify_data_path:
                    stem = Path(rel_path).stem
                    id2path[stem] = sub_dir
                    continue

                if os.path.exists(os.path.join(self.cfg.root, sub_dir)):    
                    stem = Path(rel_path).stem
                    id2path[stem] = sub_dir
                else:
                    print(os.path.exists(os.path.join(self.cfg.root, sub_dir)))
                    exists += 1

        print(f'number of files not exists: {exists}')
        return id2path

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        path = self.filenames[idx]
        row = self.metadata_dict[path]
        img = Image.open(path).convert("RGB")
        target = row.get(self.label_col_names).to_numpy(dtype="float32")
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        # get the text/caption
        caps, cap_len, sent = self.get_caption(path)
        # example call:  text_encoder(**caps).last_hidden_state[:, 0, :] 

        meta = {
            "dicom_id": row.get("dicom_id"),
            "study_id": row.get("study_id"),
            "subject_id": row.get("subject_id"),
            "target": target,
            "path": path,
            "image": img,
            "caption": caps,
            "caption_len": cap_len,
            "caption_raw": sent
        }

        return meta

    # NOTE: the following are implementations for MGCA


    def load_text_data(self, split):
        print('Get study to captions mapping...')
        filepath = os.path.join(
            MIMIC_CXR_METADATA_DIR, "captions.pickle")
        if not os.path.isfile(filepath):
            print(
                f"Caption file {filepath} does not exit. Creating captions...")
            path2sent = self.create_path_2_sent_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f)

        # filter studies to use for current split
        filenames = []
        for row in self.master_df.itertuples():
            cur_split = getattr(row, MIMIC_CXR_SPLIT_COL)
            path = getattr(row, MIMIC_CXR_PATH_COL)
            if cur_split == split and path in path2sent:
                filenames.append(path)

        return filenames, path2sent

    def create_path_2_sent_mapping(self):
        sent_lens, num_sents = [], []
        path2sent = {}
        # iterrows is not faster than itertuples ...  but it is ok
        for _, row in tqdm(self.master_df.iterrows(), total=self.master_df.shape[0]):
            # pick impression, findings, last_paragraph
            captions = ""
            captions += row["impression"]
            captions += " "
            captions += row["findings"]

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:
                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())
                # NOTE: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)

                if len(included_tokens) > 0:
                    study_sent.append(" ".join(included_tokens))

                cnt += len(included_tokens)

            if cnt >= 3:
                sent_lens.append(cnt)
                num_sents.append(len(study_sent))
                path2sent[row[MIMIC_CXR_PATH_COL]] = study_sent

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)

        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent

    def get_caption(self, path):
        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.caption_max_len
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len, sent

    def preprocess_mimic_csv_files(self, seed=42, extract_text=False):
        if not self.cfg.override_master_csv and os.path.exists(MIMIC_CXR_MASTER_CSV):
            print('Retrieval preprocessed master.csv file...')
            return pd.read_csv(MIMIC_CXR_MASTER_CSV)

        print('Creating master csv file for mimic...')
        np.random.seed(seed)
        if extract_text:
            extract_mimic_text()
        metadata_df = pd.read_csv(MIMIC_CXR_META_CSV, compression="infer")
        metadata_df = metadata_df[["dicom_id", "subject_id",
                                "study_id", "ViewPosition"]].astype(str)
        metadata_df["study_id"] = metadata_df["study_id"].apply(lambda x: "s"+x)
        # Only keep frontal images
        metadata_df = metadata_df[metadata_df["ViewPosition"].isin(["PA", "AP"])]

        text_df = pd.read_csv(MIMIC_CXR_TEXT_CSV)
        text_df.dropna(subset=["impression", "findings"], how="all", inplace=True)
        text_df = text_df[["study", "impression", "findings"]]
        text_df.rename(columns={"study": "study_id"}, inplace=True)

        split_df = pd.read_csv(MIMIC_CXR_SPLIT_CSV, compression="infer")
        split_df = split_df.astype(str)
        split_df["study_id"] = split_df["study_id"].apply(lambda x: "s"+x)

        # TODO: merge validate and test into test.
        split_df["split"] = split_df["split"].apply(
            lambda x: "valid" if x == "validate" else x)

        chexpert_df = pd.read_csv(MIMIC_CXR_CHEXPERT_CSV, compression="infer")
        chexpert_df[["subject_id", "study_id"]] = chexpert_df[[
            "subject_id", "study_id"]].astype(str)
        chexpert_df["study_id"] = chexpert_df["study_id"].apply(lambda x: "s"+x)

        master_df = pd.merge(metadata_df, text_df, on="study_id", how="left")
        master_df = pd.merge(master_df, split_df, on=["dicom_id", "subject_id", "study_id"], how="inner")
        master_df.dropna(subset=["impression", "findings"], how="all", inplace=True)
        
        n = len(master_df)
        master_data = master_df.values
        root_dir = MIMIC_CXR_JPG_DATA_DIR
        path_list = []
        for i in range(n):
            row = master_data[i]
            file_path = "%s/p%s/p%s/%s/%s.jpg" % (root_dir, str(
                row[1])[:2], str(row[1]), str(row[2]), str(row[0]))
            path_list.append(file_path)
            
        master_df.insert(loc=0, column="Path", value=path_list)

        # Create labeled data df
        labeled_data_df = pd.merge(master_df, chexpert_df, on=[
                                "subject_id", "study_id"], how="inner")
        labeled_data_df.drop(["dicom_id", "subject_id", "study_id",
                            "impression", "findings"], axis=1, inplace=True)

        train_df = labeled_data_df.loc[labeled_data_df["split"] == "train"]
        train_df.to_csv(MIMIC_CXR_TRAIN_CSV, index=False)
        valid_df = labeled_data_df.loc[labeled_data_df["split"] == "valid"]
        valid_df.to_csv(MIMIC_CXR_TEST_CSV, index=False)

        # master_df.drop(["dicom_id", "subject_id", "study_id"],
        #                axis=1, inplace=True)

        # Fill nan in text
        master_df[["impression"]] = master_df[["impression"]].fillna(" ")
        master_df[["findings"]] = master_df[["findings"]].fillna(" ")
        master_df.to_csv(MIMIC_CXR_MASTER_CSV, index=False)