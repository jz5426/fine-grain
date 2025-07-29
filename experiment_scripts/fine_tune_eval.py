
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from models.vlm_models import cxrclip_model, LinearProjectionHead, mgca_model
from data.preprocess_mimic_cxr_jpg import MIMICCXRDataloader, MIMICCXRConfig
from data.preprocess_rexerr import RexErrDataloader

import pickle
from types import SimpleNamespace
import argparse
import pandas as pd
from datetime import datetime
import joblib

def l2norm(t):
    return F.normalize(t, dim = -1)

def get_dataloader(level, split, study_level_sampling, transform, cached_file_path):

    if os.path.exists(cached_file_path):
        with open(cached_file_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

    dataset = RexErrDataloader(
        f'/cluster/projects/mcintoshgroup/publicData/rexerr-v1/ReXErr-{level}-level/ReXErr-{level}-level_{split}.csv',
        '/cluster/projects/mcintoshgroup/publicData/MIMIC-CXR/MIMIC-CXR-JPG',
        study_level_sampling=study_level_sampling,
        transform=transform
    )

    # Ensure the parent directories exist
    os.makedirs(os.path.dirname(cached_file_path), exist_ok=True)
    
    # Save the object to the specified path
    with open(cached_file_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    return dataset


# -------------------------
# Cache image and text embeddings
# -------------------------
def encode_dataset(dataloader, models, pickle_dest, device):

    if os.path.exists(pickle_dest):
        try:
            encoded_data = joblib.load(pickle_dest, mmap_mode='r')
            print("[SUCCESS] Object loaded successfully.")
            return encoded_data
        except Exception as e:
            print("[ERROR] Failed to load object.")

    image_encoder = models['image_encoder']
    image_projector = models['image_projector']
    encoded_data = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # check preprocess_data.py
            tensor_images = batch['image']
            study_id = batch['study_id'] # key of the results
            labels = batch['target']

            # Encode and project the image featuress
            tensor_images = tensor_images.to(device)
            img_feats = image_encoder(tensor_images)  # shape: [N, D]
            img_feats = image_projector(img_feats)    # apply projector: shape: [N, D']

            # normalize the feature vectors
            img_feats = l2norm(img_feats)

            # individual feature/study as a dictionary in the dict
            for i, _id in enumerate(study_id):
                encoded_data.append(SimpleNamespace(**{
                    'study_id': _id,
                    'image_feats': img_feats[i].cpu(), # [512]
                    'label': labels[i].cpu()
                }))
            
            # #NOTE: temporary for testing
            # if len(encoded_data) >= 128:
            #     return encoded_data

    # Ensure the parent directories exist
    os.makedirs(os.path.dirname(pickle_dest), exist_ok=True)
    joblib.dump(encoded_data, pickle_dest, compress=3)

    return encoded_data

# -------------------------
# Training loop
# -------------------------

def train_classifier(train_x_tensor, train_y_tensor, val_x_tensor, val_y_tensor, classifier, criterion, optimizer, device, mask_uncertain_labels, patience=5, max_epochs=50, batch_size=32):
    # Create batched DataLoader
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(val_x_tensor, val_y_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        classifier.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()

            optimizer.zero_grad()
            logits = classifier(batch_x).squeeze()
            loss = criterion(logits, batch_y)

            # Mask out uncertain labels (-1.0)
            if mask_uncertain_labels:
                mask = (batch_y != -1.0).float()  # shape: (B, C), 1.0 where label is 0 or 1, 0.0 where -1
                loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)  # avoid divide-by-zero

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        classifier.eval()
        with torch.no_grad():
            val_loss = 0
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).float()

                val_logits = classifier(batch_x).squeeze()

                # current batch loss
                loss = criterion(val_logits, batch_y)

                # Mask out uncertain labels (-1.0)
                if mask_uncertain_labels:
                    mask = (batch_y != -1.0).float()  # shape: (B, C), 1.0 where label is 0 or 1, 0.0 where -1
                    loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)  # avoid divide-by-zero

                val_loss += loss

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss.item():.4f}")

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = classifier.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

    classifier.load_state_dict(best_state)

def str2bool(v):
    return v.lower() in ('true', '1', 'yes')

def parse_args():
    parser = argparse.ArgumentParser(description="Training script with configurable parameters")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-2, help="Learning rate")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience") # 100
    parser.add_argument("--epochs", type=int, default=800, help="Number of training epochs") # 800
    parser.add_argument("--prediction_threshold", type=float, default=0.5, help="Threshold for binary classification")
    parser.add_argument("--model", type=str, default="mgca_resnet_50.ckpt", help="pretrained model checkpoint file name")
    parser.add_argument("--encode_data_only", type=str2bool, default=False, help="is encode data only")
    parser.add_argument("--verify_data_path", type=str2bool, default=False, help="verify data paths")
    parser.add_argument("--mask_uncertain_labels", type=str2bool, default=True, help="mask chestpert labels (-1)")
    return parser.parse_args()

def main():

    args = parse_args()
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    PATIENCE = args.patience
    EPOCHS = args.epochs
    PREDICTION_THRESHOLD = args.prediction_threshold
    MODEL_CHECKPOINT_NAME = args.model
    EXPERIMENT_MODEL = None
    INPUT_SIZE = None
    CACHE_PARENT_DIR = None
    MODEL_NAME = None

    print("Script Parameters:")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  LEARNING_RATE: {LEARNING_RATE}")
    print(f"  PATIENCE: {PATIENCE}")
    print(f"  EPOCHS: {EPOCHS}")
    print(f"  PREDICTION_THRESHOLD: {PREDICTION_THRESHOLD}")
    print(f"  IS_VERIFY_DATA_PATH: {args.verify_data_path}")
    print(f"  IS_ENCODE_DATA_ONLY: {args.encode_data_only}")
    print(f"  IS_MASK_UNCERTAIN_LABELS: {args.mask_uncertain_labels}")

    if MODEL_CHECKPOINT_NAME in ['r50_mcc.tar', 'r50_mc.tar', 'r50_m.tar']:
        vlm = cxrclip_model(
            f'/cluster/projects/mcintoshgroup/publicData/fine-grain/CXR-CLIP-Image-Encoder/{MODEL_CHECKPOINT_NAME}', 
            '/cluster/projects/mcintoshgroup/publicData/fine-grain/CXR-CLIP-Text-Encoder/', 
            '/cluster/projects/mcintoshgroup/publicData/fine-grain/CXR-CLIP-Text-Encoder/'
            )
        if MODEL_CHECKPOINT_NAME == 'r50_mcc.tar':
            EXPERIMENT_MODEL = 'cxrclip_r50mcc'
        elif MODEL_CHECKPOINT_NAME == 'r50_mc.tar':
            EXPERIMENT_MODEL = 'cxrclip_r50mc'
        elif MODEL_CHECKPOINT_NAME == 'r50_m.tar':
            EXPERIMENT_MODEL = 'cxrclip_r50m'
        else:
            assert False

        INPUT_SIZE = 224
        MODEL_NAME = 'cxrclip'
        CACHE_PARENT_DIR = 'cxrclip_encoder_features'
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = Compose([
            Resize((INPUT_SIZE, INPUT_SIZE)),
            CenterCrop((INPUT_SIZE, INPUT_SIZE)),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
    elif MODEL_CHECKPOINT_NAME == 'mgca_resnet_50.ckpt':
        vlm = mgca_model(
            f'/cluster/projects/mcintoshgroup/publicData/fine-grain/MGCA-Image-Encoder/{MODEL_CHECKPOINT_NAME}', 
            '/cluster/projects/mcintoshgroup/publicData/fine-grain/CXR-CLIP-Text-Encoder/', 
            '/cluster/projects/mcintoshgroup/publicData/fine-grain/CXR-CLIP-Text-Encoder/'
        )
        EXPERIMENT_MODEL = 'mgca_res50'
        INPUT_SIZE = 224
        MODEL_NAME = 'mgca'
        CACHE_PARENT_DIR = 'mgca_encoder_features'
        mean = [0.5, 0.5, 0.5] 
        std = [0.5, 0.5, 0.5]
        transform = Compose([
            CenterCrop((INPUT_SIZE, INPUT_SIZE)),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])

    print(f"  INPUT_SIZE: {INPUT_SIZE}")
    print(f"  EXPERIMENT_MODEL: {EXPERIMENT_MODEL}")
    print(f"  MODEL_NAME: {MODEL_NAME}")
    print(f"  CACHE_PARENT_DIR: {CACHE_PARENT_DIR}")
    print(f"  mean: {mean}")
    print(f"  std: {std}")

    image_encoder = vlm.image_encoder
    image_projector = vlm.image_projection
    text_encoder = vlm.text_encoder
    text_projector = vlm.text_projection
    tokenizer = vlm.tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_encoder.to(device)
    image_projector.to(device)
    text_encoder.to(device)
    text_projector.to(device)

    image_encoder.eval()
    image_projector.eval()
    text_encoder.eval()
    text_projector.eval()

    # load dataset
    cfg = MIMICCXRConfig(
        root='/cluster/projects/mcintoshgroup/publicData/MIMIC-CXR/MIMIC-CXR-JPG',
        metadata_csv="/cluster/projects/mcintoshgroup/publicData/fine-grain/MIMIC-CXR-max-metadata/mimic-cxr-2.0.0-metadata.csv.gz",
        split_csv="/cluster/projects/mcintoshgroup/publicData/fine-grain/MIMIC-CXR-max-metadata/mimic-cxr-2.0.0-split.csv.gz",
        image_filenames_txt="/cluster/projects/mcintoshgroup/publicData/fine-grain/MIMIC-CXR-max-metadata/IMAGE_FILENAMES",
        label_col=None,
        chexpert_csv='/cluster/projects/mcintoshgroup/publicData/fine-grain/MIMIC-CXR-max-metadata/mimic-cxr-2.0.0-chexpert.csv.gz',
        transform=transform,
        target_transform=None,
        verify_data_path=args.verify_data_path,
        mask_uncertain_labels=args.mask_uncertain_labels,
        override_master_csv=False,
        caption_max_len=128
    )
    print('processing training split...')
    train_dataset = MIMICCXRDataloader(cfg, tokenizer, split="train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print('processing validation split...')
    val_dataset = MIMICCXRDataloader(cfg, tokenizer, split="val")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print('processing testing split...')
    test_dataset = MIMICCXRDataloader(cfg, tokenizer, split="test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Classes:", train_dataset.classes)
    print("Number of training samples:", len(train_dataset))
    print("Number of validation samples:", len(val_dataset))
    print("Number of testing samples:", len(test_dataset))

    if args.verify_data_path:
        print('finished verifying data path')
        return

    models = {
        'image_encoder': image_encoder,
        'text_encoder': text_encoder,
        'image_projector': image_projector,
        'text_projector': text_projector,
        'model_name': MODEL_NAME
    }

    print("Encoding train set...")
    encoded_train_data = encode_dataset(
        train_loader, 
        models, 
        pickle_dest=f'/cluster/projects/mcintoshgroup/publicData/fine-grain/cache/fine_tune_mimic/{CACHE_PARENT_DIR}/train_features.joblib',
        device=device)
    
    print("Encoding val set...")
    encoded_val_data = encode_dataset(
        val_loader, 
        models, 
        pickle_dest=f'/cluster/projects/mcintoshgroup/publicData/fine-grain/cache/fine_tune_mimic/{CACHE_PARENT_DIR}/val_features.joblib',
        device=device)

    print("Encoding test set...")
    encoded_test_data = encode_dataset(
        test_loader, 
        models, 
        pickle_dest=f'/cluster/projects/mcintoshgroup/publicData/fine-grain/cache/fine_tune_mimic/{CACHE_PARENT_DIR}/test_features.joblib',
        device=device)

    if args.encode_data_only:
        print('finished encoding data')
        return

    # -------------------------
    # Get data for linear projector training
    # -------------------------

    # sample the same set of indices from both train_tr and train_err for images and text
    train_img_feats = [data.image_feats for data in encoded_train_data]

    # get only the image and text features from the val data
    val_img_feats = [data.image_feats for data in encoded_val_data]

    # get only the image and text features from the test data
    test_img_feats = [data.image_feats for data in encoded_test_data]

    # combine tensors for train
    train_img_tensor = torch.stack(train_img_feats).to(device) # shape [N, D]
    train_y = torch.stack([data.label for data in encoded_train_data]).to(device)
    print(f' total training size {len(encoded_train_data)}')

    # combine tensors for train
    val_img_tensor = torch.stack(val_img_feats).to(device) # shape [N, D]
    val_y = torch.stack([data.label for data in encoded_val_data]).to(device)

    # combine tensors for test
    test_img_tensor = torch.stack(test_img_feats).to(device)  # shape [N, D]
    test_y = torch.stack([data.label for data in encoded_test_data]).to(device)

    # -------------------------
    # Define classifier
    # -------------------------
    input_dim = train_img_feats[0].shape[0]
    train_feats = train_img_tensor
    val_feats = val_img_tensor
    test_feats = test_img_tensor

    # multi-label classification training
    classifier = LinearProjectionHead(input_dim, train_y.shape[-1]).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='mean' if not args.mask_uncertain_labels else 'none')
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

    print("Training classifier...")
    train_classifier(
        train_feats, 
        train_y, 
        val_feats, 
        val_y,  
        classifier,
        criterion,
        optimizer,
        device,
        args.mask_uncertain_labels,
        patience=PATIENCE, 
        max_epochs=EPOCHS, 
        batch_size=BATCH_SIZE)

    # -------------------------
    # Evaluation on test set
    # -------------------------

    classifier.eval()
    test_dataset = TensorDataset(test_feats, test_y)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_probs, all_logits, all_labels = [], [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)  # shape: (B, C), values in {1.0, 0.0, -1.0}

            logits = classifier(batch_x)
            all_logits.append(logits.cpu())
            all_labels.append(batch_y.cpu())

    # Stack tensors
    all_logits = torch.cat(all_logits)      # shape (N, C)
    all_labels = torch.cat(all_labels)      # shape (N, C)

    # Apply sigmoid to get predicted probabilities
    all_probs = torch.sigmoid(all_logits)

    # Create mask to ignore uncertain labels (-1.0)
    mask = (all_labels != -1.0)

    # Get masked values
    flat_labels = all_labels[mask].numpy()
    flat_probs = all_probs[mask].numpy()
    flat_preds = (all_probs >= PREDICTION_THRESHOLD)[mask].numpy()

    # Compute masked micro accuracy
    accuracy = (flat_preds == flat_labels).mean()

    # Compute masked micro PR-AUC
    prauc = average_precision_score(flat_labels, flat_probs)

    # Compute AUC (micro)
    auc = roc_auc_score(flat_labels, flat_probs, average="micro")

    print(f"Masked Micro Accuracy: {accuracy:.4f}")
    print(f"Masked Micro PR-AUC:  {prauc:.4f}")
    print(f"Masked AUC:  {auc:.4f}")


    # -------------------------
    # save results to spreadsheet
    # -------------------------

    # Create a result dictionary
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "batch_size": BATCH_SIZE,
        "input_size": INPUT_SIZE,
        "learning_rate": LEARNING_RATE,
        "patience": PATIENCE,
        "epochs": EPOCHS,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "test_accuracy": accuracy,
        "pr_auc": prauc,
        'auc': auc
    }

    # Convert to a DataFrame
    results_df = pd.DataFrame([results])

    # Define output path
    out_path = f"/cluster/projects/mcintoshgroup/publicData/fine-grain/experiment/fine_tune_mimic/{EXPERIMENT_MODEL}_results.csv"

    # If file exists, append; otherwise create new
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        results_df.to_csv(out_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(out_path, index=False)

    print(f"Results saved to {out_path}")

if __name__ == '__main__':

    main()