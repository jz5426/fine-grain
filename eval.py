"""
1. load the model weights for the text and report.
2. load the data class
3. cache the embeddings first
4. few-shot classification of whether they are true pairs.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm
import numpy as np

from models import cxrclip_model
from preprocess_data import RexErrDataset

import os
import pickle

def get_dataset(level, split, study_level_sampling, transform, cached_file_path):

    if os.path.exists(cached_file_path):
        with open(cached_file_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

    dataset = RexErrDataset(
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
def encode_dataset(dataloader, models):
    image_encoder = models['image_encoder']
    text_encoder = models['text_encoder']
    tokenizer = models['tokenizer']
    image_projector = models['image_projector']
    text_projector = models['text_projector']

    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # check preprocess_data.py
            tensor_images = batch['tensor_images'][0]
            origin_text = batch['original_text']    # assume this is the input text
            err_text = batch['error_text']
            study_id = batch['study_id'] # key of the results

            # Encode and project the image featuress
            tensor_images = tensor_images.to(device)
            img_feats = image_encoder(tensor_images)  # shape: [N, D]
            img_feats = image_projector(img_feats)    # apply projector: shape: [N, D']

            # Encode original text
            inputs = tokenizer(origin_text, return_tensors="pt", padding=True, truncation=True).to(device)
            origin_txt_feats = text_encoder(**inputs).last_hidden_state[:, 0, :]  # CLS token
            origin_txt_feats = text_projector(origin_txt_feats)

            # Encoder error text
            inputs = tokenizer(err_text, return_tensors="pt", padding=True, truncation=True).to(device)
            err_txt_feats = text_encoder(**inputs).last_hidden_state[:, 0, :]  # CLS token
            err_txt_feats = text_projector(err_txt_feats)

            assert img_feats.shape == origin_txt_feats.shape and origin_txt_feats.shape == err_txt_feats.shape

            results.append({
                'study_id': study_id,
                'origin_txt_feats': origin_txt_feats,
                'error_txt_feats': err_txt_feats,
                'image_feats': img_feats
            })

    return results


# -------------------------
# Training loop
# -------------------------
def train_classifier(train_x, train_y, val_x, val_y, patience=5, max_epochs=50):
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        classifier.train()
        optimizer.zero_grad()

        logits = classifier(train_x.to(device)).squeeze()
        loss = criterion(logits, train_y.to(device).float())
        loss.backward()
        optimizer.step()

        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_x.to(device)).squeeze()
            val_loss = criterion(val_logits, val_y.to(device).float())

        print(f"Epoch {epoch} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

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


if __name__ == '__main__':

    #TODO: define the numbers as hyperparameters
    #TODO: revise the following code to suit our task.

    cxrclip = cxrclip_model(
        '/cluster/projects/mcintoshgroup/publicData/fine-grain/CXR-CLIP-Image-Encoder/r50_m.tar', 
        '/cluster/projects/mcintoshgroup/publicData/fine-grain/CXR-CLIP-Text-Encoder/', 
        '/cluster/projects/mcintoshgroup/publicData/fine-grain/CXR-CLIP-Text-Encoder'
        )
    image_encoder = cxrclip.image_encoder
    image_projector = cxrclip.image_projection
    text_encoder = cxrclip.text_encoder
    text_projector = cxrclip.text_projection
    tokenizer = cxrclip.tokenizer

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
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]) # TODO: check with CXR-CLIP
    ])
    train_dataset = get_dataset('report', 'train', False, transform, '/cluster/projects/mcintoshgroup/publicData/fine-grain/rexerr_train.pkl')
    val_dataset = get_dataset('report', 'val', False, transform, '/cluster/projects/mcintoshgroup/publicData/fine-grain/rexerr_val.pkl')
    test_dataset = get_dataset('report', 'test', False, transform, '/cluster/projects/mcintoshgroup/publicData/fine-grain/rexerr_test.pkl')

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    models = {
        'image_encoder': image_encoder,
        'text_encoder': text_encoder,
        'tokenizer': tokenizer,
        'image_projector': image_projector,
        'text_projector': text_projector
    }

    print("Encoding train set...")
    train_img, train_txt, train_lbl = encode_dataset(train_loader, models)

    print("Encoding val set...")
    val_img, val_txt, val_lbl = encode_dataset(val_loader, models)

    print("Encoding test set...")
    test_img, test_txt, test_lbl = encode_dataset(test_loader, models)

    # -------------------------
    # Define classifier
    # -------------------------
    input_dim = train_img.shape[1] + train_txt.shape[1] # TODO: concatenation or subtraction or addition
    classifier = nn.Sequential(
        nn.Linear(input_dim, 1)
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)

    print("Training classifier...")
    train_feats = torch.cat([train_img, train_txt], dim=1)
    val_feats = torch.cat([val_img, val_txt], dim=1)
    train_classifier(train_feats, train_lbl, val_feats, val_lbl)

    # -------------------------
    # Evaluation on test set
    # -------------------------
    classifier.eval()
    with torch.no_grad():
        test_feats = torch.cat([test_img, test_txt], dim=1).to(device)
        test_logits = classifier(test_feats).squeeze()
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_preds = (test_probs >= 0.5).astype(int)
        test_labels = test_lbl.numpy()

        acc = np.mean(test_preds == test_labels)
        print(f"Test Accuracy: {acc:.4f}")
