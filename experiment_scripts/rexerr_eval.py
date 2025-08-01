"""
1. load the model weights for the text and report.
2. load the data class
3. cache the embeddings first
4. few-shot classification of whether they are true pairs.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score

from models.vlm_models import cxrclip_model, LinearProjectionHead, mgca_model
from rexerr_dataloader import RexErrDataloader

import os
import pickle
import random
from types import SimpleNamespace
import argparse
import pandas as pd
from datetime import datetime
import joblib

def l2norm(t):
    return F.normalize(t, dim = -1)

def get_dataset(level, split, study_level_sampling, transform, cached_file_path):

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
def encode_dataset(dataloader, models, pickle_dest, tokens_max_length=256):

    if os.path.exists(pickle_dest):
        try:
            data = joblib.load(pickle_dest, mmap_mode='r')
            print("[SUCCESS] Object loaded successfully.")
            ground_truth_pairs, err_pairs = data
            return ground_truth_pairs, err_pairs
        except Exception as e:
            print("[ERROR] Failed to load object.")

    image_encoder = models['image_encoder']
    text_encoder = models['text_encoder']
    tokenizer = models['tokenizer']
    image_projector = models['image_projector']
    text_projector = models['text_projector']
    model_name = models['model_name']

    ground_truth_pairs, err_pairs = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # check preprocess_data.py
            tensor_images = batch['tensor_images'][0]
            origin_text = batch['caption']    # assume this is the input text
            err_text = batch['error_caption']
            study_id = batch['study_id'] # key of the results

            # Encode and project the image featuress
            tensor_images = tensor_images.to(device)
            img_feats = image_encoder(tensor_images)  # shape: [N, D]
            img_feats = image_projector(img_feats)    # apply projector: shape: [N, D']

            # Encode original text
            inputs = tokenizer(origin_text, return_tensors="pt", padding=True, truncation=True, max_length=tokens_max_length).to(device)
            if model_name == 'cxrclip':
                origin_txt_feats = text_encoder(**inputs).last_hidden_state[:, 0, :]  # CLS token
            elif model_name == 'mgca':
                origin_txt_feats, word_feat_q, word_attn_q, sents = text_encoder(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])

            origin_txt_feats = text_projector(origin_txt_feats)

            # Encoder error text
            inputs = tokenizer(err_text, return_tensors="pt", padding=True, truncation=True, max_length=tokens_max_length).to(device)
            if model_name == 'cxrclip':
                err_txt_feats = text_encoder(**inputs).last_hidden_state[:, 0, :]  # CLS token
            elif model_name == 'mgca':
                err_txt_feats, word_feat_q, word_attn_q, sents = text_encoder(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'],)
            
            err_txt_feats = text_projector(err_txt_feats)

            assert img_feats.shape == origin_txt_feats.shape and origin_txt_feats.shape == err_txt_feats.shape

            # normalize the feature vectors
            img_feats = l2norm(img_feats)
            origin_txt_feats = l2norm(origin_txt_feats)
            err_txt_feats = l2norm(err_txt_feats)

            # individual feature/study as a dictionary in the dict
            for i, _id in enumerate(study_id):
                ground_truth_pairs.append(SimpleNamespace(**{
                    'study_id': _id,
                    'text_feats': origin_txt_feats[i].cpu(), # [512]
                    'image_feats': img_feats[i].cpu(), # [512]
                    'label': 1
                }))

                err_pairs.append(SimpleNamespace(**{
                    'study_id': _id,
                    'text_feats': err_txt_feats[i].cpu(),
                    'image_feats': img_feats[i].cpu(),
                    'label': 0
                }))
            
            # #NOTE: temporary
            # if len(ground_truth_pairs) >= 256:
            #     return ground_truth_pairs, err_pairs

    # Ensure the parent directories exist
    os.makedirs(os.path.dirname(pickle_dest), exist_ok=True)
    joblib.dump((ground_truth_pairs, err_pairs), pickle_dest, compress=3)

    return ground_truth_pairs, err_pairs

# -------------------------
# Training loop
# -------------------------

def train_classifier(train_x_tensor, train_y_tensor, val_x_tensor, val_y_tensor, patience=5, max_epochs=50, batch_size=32):
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
                val_loss += criterion(val_logits, batch_y)

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

def parse_args():
    parser = argparse.ArgumentParser(description="Training script with configurable parameters")

    parser.add_argument("--few_shot", type=float, default=0.01, help="Few-shot learning ratio")
    parser.add_argument("--fusion_type", type=str, default="text_only", choices=["concatenate", "subtraction", "addition", "text_only"], help="Type of fusion method")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-2, help="Learning rate")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience") # 100
    parser.add_argument("--epochs", type=int, default=800, help="Number of training epochs") # 800
    parser.add_argument("--prediction_threshold", type=float, default=0.5, help="Threshold for binary classification")
    parser.add_argument("--model", type=str, default="mgca_resnet_50.ckpt", help="pretrained model checkpoint file name")
    parser.add_argument("--text_max_length", type=int, default=1024, help="maximum number of tokens")

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    FEW_SHOT = args.few_shot
    FUSION_TYPE = args.fusion_type
    BATCH_SIZE = args.batch_size
    
    LEARNING_RATE = args.learning_rate
    PATIENCE = args.patience
    EPOCHS = args.epochs
    PREDICTION_THRESHOLD = args.prediction_threshold
    MODEL_CHECKPOINT_NAME = args.model
    IS_TEXT_ONLY_EVALUATION = True if args.fusion_type =='text_only' else False
    TEXT_TRUNCATION = args.text_max_length
    EXPERIMENT_MODEL = None
    INPUT_SIZE = None
    CACHE_PARENT_DIR = None
    MODEL_NAME = None

    print("Script Parameters:")
    print(f"  FEW_SHOT: {FEW_SHOT}")
    print(f"  FUSION_TYPE: {FUSION_TYPE}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  LEARNING_RATE: {LEARNING_RATE}")
    print(f"  PATIENCE: {PATIENCE}")
    print(f"  EPOCHS: {EPOCHS}")
    print(f"  PREDICTION_THRESHOLD: {PREDICTION_THRESHOLD}")
    print(f"  TEXT_TRUNCATION: {TEXT_TRUNCATION}")


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
        # TEXT_TRUNCATION = 256
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
        # TEXT_TRUNCATION = 256
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
    transform = Compose([
        Resize((INPUT_SIZE, INPUT_SIZE)),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    train_dataset = get_dataset('report', 'train', False, transform, f'/cluster/projects/mcintoshgroup/publicData/fine-grain/cache/rexerr_train.pkl')
    val_dataset = get_dataset('report', 'val', False, transform, f'/cluster/projects/mcintoshgroup/publicData/fine-grain/cache/rexerr_val.pkl')
    test_dataset = get_dataset('report', 'test', False, transform, f'/cluster/projects/mcintoshgroup/publicData/fine-grain/cache/rexerr_test.pkl')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    models = {
        'image_encoder': image_encoder,
        'text_encoder': text_encoder,
        'tokenizer': tokenizer,
        'image_projector': image_projector,
        'text_projector': text_projector,
        'model_name': MODEL_NAME
    }

    print("Encoding train set...")
    train_gt, train_err = encode_dataset(
        train_loader, 
        models, 
        pickle_dest=f'/cluster/projects/mcintoshgroup/publicData/fine-grain/cache/{CACHE_PARENT_DIR}_{TEXT_TRUNCATION}/train_features.joblib', 
        tokens_max_length=TEXT_TRUNCATION)
    
    print("Encoding val set...")
    val_gt, val_err = encode_dataset(
        val_loader, 
        models, 
        pickle_dest=f'/cluster/projects/mcintoshgroup/publicData/fine-grain/cache/{CACHE_PARENT_DIR}_{TEXT_TRUNCATION}/val_features.joblib',
        tokens_max_length=TEXT_TRUNCATION)

    print("Encoding test set...")
    test_gt, test_err = encode_dataset(
        test_loader, 
        models, 
        pickle_dest=f'/cluster/projects/mcintoshgroup/publicData/fine-grain/cache/{CACHE_PARENT_DIR}_{TEXT_TRUNCATION}/test_features.joblib',
        tokens_max_length=TEXT_TRUNCATION)

    # -------------------------
    # Few-shot sampling (by %) and combine the tensors
    # -------------------------

    sample_size = int((len(train_gt) * FEW_SHOT) // 2)
    sampled_indices = random.sample(range(len(train_gt)), sample_size)

    # sample the same set of indices from both train_tr and train_err for images and text
    sampled_train_gt_img = [train_gt[i].image_feats for i in sampled_indices]
    sampled_train_gt_txt = [train_gt[i].text_feats for i in sampled_indices]

    sampled_train_err_img = [train_err[i].image_feats for i in sampled_indices]
    sampled_train_err_txt = [train_err[i].text_feats for i in sampled_indices]

    # get only the image and text features from the val data
    val_gt_img = [data.image_feats for data in val_gt]
    val_gt_txt = [data.text_feats for data in val_gt]
    val_err_img = [data.image_feats for data in val_err]
    val_err_txt = [data.text_feats for data in val_err]

    # get only the image and text features from the test data
    test_gt_img = [data.image_feats for data in test_gt]
    test_gt_txt = [data.text_feats for data in test_gt]
    test_err_img = [data.image_feats for data in test_err]
    test_err_txt = [data.text_feats for data in test_err]

    # combine tensors for train
    train_gt_img_tensor = torch.stack(sampled_train_gt_img).to(device) # shape [N, D]
    train_gt_txt_tensor = torch.stack(sampled_train_gt_txt).to(device) # shape [N, D]
    train_err_img_tensor = torch.stack(sampled_train_err_img).to(device) # shape [N, D]
    train_err_txt_tensor = torch.stack(sampled_train_err_txt).to(device) # shape [N, D]
    # combine the image tensor (NOTE: it should be duplicate of each other) as a single training set
    train_img_tensor = torch.concat((train_gt_img_tensor, train_err_img_tensor), dim=0) # [2N, D]
    train_txt_tensor = torch.concat((train_gt_txt_tensor, train_err_txt_tensor), dim=0) # [2N, D]
    train_y= torch.tensor([1]*train_gt_img_tensor.shape[0]+[0]*train_err_img_tensor.shape[0]).to(device)  # shape [N+N], N is number of samples for each 
    print(f' Few-shot size: {train_y.shape}, total training size {len(train_gt)}')

    # combine tensors for train
    val_gt_img_tensor = torch.stack(val_gt_img).to(device) # shape [N, D]
    val_gt_txt_tensor = torch.stack(val_gt_txt).to(device) # shape [N, D]
    val_err_img_tensor = torch.stack(val_err_img).to(device) # shape [N, D]
    val_err_txt_tensor = torch.stack(val_err_txt).to(device) # shape [N, D]
    # combine the image tensor (NOTE: it should be duplicate of each other) as a single training set
    val_img_tensor = torch.concat((val_gt_img_tensor, val_err_img_tensor), dim=0)
    val_txt_tensor = torch.concat((val_gt_txt_tensor, val_err_txt_tensor), dim=0)
    val_y = torch.tensor([1]*val_gt_img_tensor.shape[0]+[0]*val_err_img_tensor.shape[0]).to(device)

    # combine tensors for test
    test_gt_img_tensor = torch.stack(test_gt_img).to(device)  # shape [N, D]
    test_gt_txt_tensor = torch.stack(test_gt_txt).to(device)  # shape [N, D]
    test_err_img_tensor = torch.stack(test_err_img).to(device)  # shape [N, D]
    test_err_txt_tensor = torch.stack(test_err_txt).to(device)  # shape [N, D]
    # combine the image tensor (NOTE: it should be duplicate of each other) as a single training set
    test_img_tensor = torch.concat((test_gt_img_tensor, test_err_img_tensor), dim=0)
    test_txt_tensor = torch.concat((test_gt_txt_tensor, test_err_txt_tensor), dim=0)
    test_y = torch.tensor([1]*test_gt_img_tensor.shape[0]+[0]*test_err_img_tensor.shape[0]).to(device)

    # -------------------------
    # Define classifier
    # -------------------------
    if FUSION_TYPE == 'concatenate':
        input_dim = sampled_train_gt_img[0].shape[0] + sampled_train_gt_txt[0].shape[0]
        train_feats = torch.concat([train_img_tensor, train_txt_tensor], dim=1)
        val_feats = torch.concat([val_img_tensor, val_txt_tensor], dim=1)
        test_feats = torch.concat([test_img_tensor, test_txt_tensor], dim=1)

    elif FUSION_TYPE == 'subtraction':
        input_dim = sampled_train_gt_img[0].shape[0]
        train_feats = train_img_tensor - train_txt_tensor
        val_feats = val_img_tensor - val_txt_tensor
        test_feats = test_img_tensor - test_txt_tensor

    elif FUSION_TYPE == 'addition':
        input_dim = sampled_train_gt_img[0].shape[0]
        train_feats = train_img_tensor + train_txt_tensor
        val_feats = val_img_tensor + val_txt_tensor
        test_feats = test_img_tensor + test_txt_tensor
    
    elif FUSION_TYPE == 'text_only':
        input_dim = train_txt_tensor[0].shape[0] # should be the same as the image embedding dimension
        train_feats = train_txt_tensor
        val_feats = val_txt_tensor
        test_feats = test_txt_tensor

    classifier = LinearProjectionHead(input_dim, 1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

    print("Training classifier...")
    train_classifier(
        train_feats, 
        train_y, 
        val_feats, 
        val_y,  
        patience=PATIENCE, 
        max_epochs=EPOCHS, 
        batch_size=BATCH_SIZE
        )

    # -------------------------
    # Evaluation on test set
    # -------------------------

    classifier.eval()
    test_dataset = TensorDataset(test_feats, test_y)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()

            test_logits = classifier(batch_x).squeeze()
            test_probs = torch.sigmoid(test_logits).cpu().numpy()
            test_preds = (test_probs >= PREDICTION_THRESHOLD).astype(int)
            test_labels = batch_y.cpu().numpy()  # <-- fix: use batch_y, not test_y

            all_probs.extend(test_probs)
            all_preds.extend(test_preds)
            all_labels.extend(test_labels)

        # Convert to NumPy arrays
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute accuracy
        accuracy = (all_preds == all_labels).mean()

        # Compute PRAUC
        prauc = average_precision_score(all_labels, all_probs)

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test PRAUC: {prauc:.4f}")

    # -------------------------
    # save results to spreadsheet
    # -------------------------

    # Create a result dictionary
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "few_shot": FEW_SHOT,
        "fusion_type": FUSION_TYPE,
        "batch_size": BATCH_SIZE,
        "input_size": INPUT_SIZE,
        "learning_rate": LEARNING_RATE,
        "patience": PATIENCE,
        "epochs": EPOCHS,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "text_max_length": TEXT_TRUNCATION,
        "test_accuracy": accuracy,
        "pr_auc": prauc
    }

    # Convert to a DataFrame
    results_df = pd.DataFrame([results])

    # Define output path
    out_path = f"/cluster/projects/mcintoshgroup/publicData/fine-grain/experiment/{EXPERIMENT_MODEL}{'_text_only' if IS_TEXT_ONLY_EVALUATION else ''}_results.csv"

    # If file exists, append; otherwise create new
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        results_df.to_csv(out_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(out_path, index=False)

    print(f"Results saved to {out_path}")