"""
1. load the model weights for the text and report.
2. load the data class
3. cache the embeddings first
4. few-shot classification of whether they are true pairs.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from models import cxrclip_model, LinearProjectionHead
from preprocess_data import RexErrDataset

import os
import pickle
import random

def l2norm(t):
    return F.normalize(t, dim = -1)

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

    ground_truth_pairs, err_pairs = [], []
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

            # normalize the feature vectors
            img_feats = l2norm(img_feats)
            origin_txt_feats = l2norm(origin_txt_feats)
            err_txt_feats = l2norm(err_txt_feats)

            ground_truth_pairs.append({
                'study_id': study_id,
                'text_feats': origin_txt_feats,
                'image_feats': img_feats,
                'label': 1
            })

            err_pairs.append({
                'study_id': study_id,
                'text_feats': err_txt_feats,
                'image_feats': img_feats,
                'label': 0
            })

    # TODO: save as pickle object.
    

    return ground_truth_pairs, err_pairs


# -------------------------
# Training loop
# -------------------------

def train_classifier(train_x_tensor, train_y_tensor, val_x_tensor, val_y_tensor, patience=5, max_epochs=50, batch_size=32):
    # Create batched DataLoader
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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
            val_logits = classifier(val_x_tensor.to(device)).squeeze()
            val_loss = criterion(val_logits, val_y_tensor.to(device).float())

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


if __name__ == '__main__':

    #TODO: define the numbers as hyperparameters
    #TODO: revise the following code to suit our task.

    FEW_SHOT = 0.1
    FUSION_TYPE = 'concatenate' # 'concatenate' 'subtraction' 'addition'text_projection_head

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
                  std=[0.229, 0.224, 0.225])
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
    train_gt, train_err = encode_dataset(train_loader, models)
    
    print("Encoding val set...")
    val_gt, val_err = encode_dataset(val_loader, models)

    print("Encoding test set...")
    test_gt, test_err = encode_dataset(test_loader, models)

    # -------------------------
    # Few-shot sampling (by %) and combine the tensors
    # -------------------------

    sample_size = int(len(train_gt) * FEW_SHOT)
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
    # combine the image tensor (NOTE: it should be duplicate of each other)
    train_img_tensor = torch.concat((train_gt_img_tensor, train_err_img_tensor), dim=0)
    train_txt_tensor = torch.concat((train_gt_txt_tensor, train_err_txt_tensor), dim=0)
    train_y= torch.tensor([1]*train_gt_img_tensor.shape[0]+[0]*train_err_img_tensor.shape[0]).to(device)  # shape [N+N], N is number of samples for each 

    # combine tensors for train
    val_gt_img_tensor = torch.stack(val_gt_img).to(device) # shape [N, D]
    val_gt_txt_tensor = torch.stack(val_gt_txt).to(device) # shape [N, D]
    val_err_img_tensor = torch.stack(val_err_img).to(device) # shape [N, D]
    val_err_txt_tensor = torch.stack(val_err_txt).to(device) # shape [N, D]
    # combine the image tensor (NOTE: it should be duplicate of each other)
    val_img_tensor = torch.concat((val_gt_img_tensor, val_err_img_tensor), dim=0)
    val_txt_tensor = torch.concat((val_gt_txt_tensor, val_err_txt_tensor), dim=0)
    val_y = torch.tensor([1]*val_gt_img_tensor.shape[0]+[0]*val_err_img_tensor.shape[0]).to(device)

    # combine tensors for test
    test_gt_img_tensor = torch.stack(test_gt_img).to(device)  # shape [N, D]
    test_gt_txt_tensor = torch.stack(test_gt_txt).to(device)  # shape [N, D]
    test_err_img_tensor = torch.stack(test_err_img).to(device)  # shape [N, D]
    test_err_txt_tensor = torch.stack(test_err_txt).to(device)  # shape [N, D]
    # combine the image tensor (NOTE: it should be duplicate of each other)
    test_img_tensor = torch.concat((test_gt_img_tensor, test_err_img_tensor), dim=0)
    test_txt_tensor = torch.concat((test_gt_txt_tensor, test_err_txt_tensor), dim=0)
    test_y = torch.tensor([1]*test_gt_img_tensor.shape[0]+[0]*test_err_img_tensor.shape[0]).to(device)

    # -------------------------
    # Define classifier
    # -------------------------
    if FUSION_TYPE == 'concatenate':
        input_dim = sampled_train_gt_img[0].shape[1] + sampled_train_gt_txt.shape[1]
        train_feats = torch.concat([train_img_tensor, train_txt_tensor], dim=1)
        val_feats = torch.concat([val_img_tensor, val_txt_tensor], dim=1)
        test_feats = torch.concat([test_img_tensor, test_txt_tensor], dim=1)

    elif FUSION_TYPE == 'subtraction':
        input_dim = sampled_train_gt_img[0].shape[1]
        train_feats = train_img_tensor - train_txt_tensor
        val_feats = val_img_tensor - val_txt_tensor
        test_feats = test_img_tensor - test_txt_tensor

    elif FUSION_TYPE == 'addition':
        input_dim = sampled_train_gt_img[0].shape[1]
        train_feats = train_img_tensor + train_txt_tensor
        val_feats = val_img_tensor + val_txt_tensor
        test_feats = test_img_tensor + test_txt_tensor

    classifier = LinearProjectionHead(input_dim, 1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)

    print("Training classifier...")
    # train_feats = torch.cat([train_img, train_txt], dim=1)
    # val_feats = torch.cat([val_img, val_txt], dim=1)
    train_classifier(train_feats, train_y, val_feats, val_y,  patience=5, max_epochs=50, batch_size=32)

    # -------------------------
    # Evaluation on test set
    # -------------------------
    classifier.eval()
    with torch.no_grad():
        # test_feats = torch.cat([test_img, test_txt], dim=1).to(device)
        test_logits = classifier(test_feats).squeeze()
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_preds = (test_probs >= 0.5).astype(int)
        test_labels = test_y.numpy()

        acc = np.mean(test_preds == test_labels)
        print(f"Test Accuracy: {acc:.4f}")
