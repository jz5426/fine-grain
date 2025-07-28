import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Literal, Dict

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from mgca_constants import *
from mgca_utils import extract_mimic_text

SplitT = Literal["train", "validate", "val", "test"]

@dataclass
class MIMICCXRConfig:
    root: str                                   # root directory that contains the "files/" tree
    metadata_csv: str                           # path to mimic-cxr-2.0.0-metadata.csv.gz
    split_csv: str                              # path to mimic-cxr-2.0.0-split.csv.gz
    image_filenames_txt: str                    # path to IMAGE_FILENAMES
    label_col: Optional[str] = None             # if None -> multi-label CheXpert; else single-label categorical
    chexpert_csv: Optional[str] = None          # path to mimic-cxr-2.0.0-chexpert.csv.gz for multi-label
    transform: Optional[Callable] = None
    target_transform: Optional[Callable] = None
    drop_missing_labels: bool = True
    verify_data_path: bool = False
    mask_uncertain_labels: bool = False

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

    def __init__(self, cfg: MIMICCXRConfig, split: SplitT = "train"):
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
        if not self.cfg.mask_uncertain_labels:
            df[label_cols] = df[label_cols].fillna(0.0).replace(-1.0, 0.0).astype("float32")
        else:
            df[label_cols] = df[label_cols].fillna(0.0).astype("float32")

        # Cache the targets as a NumPy array for speed
        self.targets = df[label_cols].to_numpy(dtype="float32")
        self.label_cols = label_cols

        self.df = df.reset_index(drop=True)
        self.transform = cfg.transform
        self.target_transform = cfg.target_transform

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
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        target = torch.from_numpy(self.targets[idx])  # float32 tensor of shape (14,)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        meta = {
            "dicom_id": row["dicom_id"],
            "study_id": row.get("study_id"),
            "subject_id": row.get("subject_id"),
            "path": row["path"],
            "image": img,
            "target": target
        }

        return meta

def preprocess_mimic_csv_files():
    extract_text = False
    np.random.seed(42)
    if extract_text:
        extract_mimic_text()
    metadata_df = pd.read_csv(MIMIC_CXR_META_CSV)
    metadata_df = metadata_df[["dicom_id", "subject_id",
                               "study_id", "ViewPosition"]].astype(str)
    metadata_df["study_id"] = metadata_df["study_id"].apply(lambda x: "s"+x)
    # Only keep frontal images
    metadata_df = metadata_df[metadata_df["ViewPosition"].isin(["PA", "AP"])]

    text_df = pd.read_csv(MIMIC_CXR_TEXT_CSV)
    text_df.dropna(subset=["impression", "findings"], how="all", inplace=True)
    text_df = text_df[["study", "impression", "findings"]]
    text_df.rename(columns={"study": "study_id"}, inplace=True)

    split_df = pd.read_csv(MIMIC_CXR_SPLIT_CSV)
    split_df = split_df.astype(str)
    split_df["study_id"] = split_df["study_id"].apply(lambda x: "s"+x)

    # TODO: merge validate and test into test.
    split_df["split"] = split_df["split"].apply(
        lambda x: "valid" if x == "validate" or x == "test" else x)

    chexpert_df = pd.read_csv(MIMIC_CXR_CHEXPERT_CSV)
    chexpert_df[["subject_id", "study_id"]] = chexpert_df[[
        "subject_id", "study_id"]].astype(str)
    chexpert_df["study_id"] = chexpert_df["study_id"].apply(lambda x: "s"+x)

    master_df = pd.merge(metadata_df, text_df, on="study_id", how="left")
    master_df = pd.merge(master_df, split_df, on=["dicom_id", "subject_id", "study_id"], how="inner")
    master_df.dropna(subset=["impression", "findings"], how="all", inplace=True)
    
    n = len(master_df)
    master_data = master_df.values

    root_dir = str(MIMIC_CXR_DATA_DIR).split("/")[-1] + "/files"
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


if __name__ == "__main__":
    main()