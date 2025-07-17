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

from models import cxrclip
from preprocess_data import RexErrDataset


def get_dataset(level, split, study_level_sampling, transform):
    return RexErrDataset(
        f'/cluster/projects/mcintoshgroup/publicData/rexerr-v1/ReXErr-{level}-level/ReXErr-{level}-level_{split}.csv',
        '/cluster/projects/mcintoshgroup/publicData/MIMIC-CXR/MIMIC-CXR-JPG',
        study_level_sampling=study_level_sampling,
        transform=transform
    )

# -------------------------
# Cache image and text embeddings
# -------------------------
def encode_dataset(dataloader):
    all_img_embeddings = []
    all_txt_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            tensor_images = [img.to(device) for img in batch['tensor_images']]
            labels = batch['label']  # assume your dataset returns this
            texts = batch['text']    # assume this is the input text

            # Encode images
            img_feats = torch.stack([image_encoder(img.unsqueeze(0)).squeeze(0) for img in tensor_images])
            # Encode text
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
            txt_feats = text_encoder(**inputs).last_hidden_state[:, 0, :]  # CLS token

            all_img_embeddings.append(img_feats.cpu())
            all_txt_embeddings.append(txt_feats.cpu())
            all_labels.append(torch.tensor(labels))

    return torch.cat(all_img_embeddings), torch.cat(all_txt_embeddings), torch.cat(all_labels)


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

    cxrclip = cxrclip('image encoder weight path', 'text encoder weight path', 'tokenizer path, should be the same as the text encoder')
    image_encoder = cxrclip.image_encoder
    text_encoder = cxrclip.text_encoder
    tokenizer = cxrclip.tokenizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_encoder.to(device)
    text_encoder.to(device)
    image_encoder.eval()
    text_encoder.eval()

    # load dataset
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    train_dataset = get_dataset('report', 'train', False, transform)
    val_dataset = get_dataset('report', 'val', False, transform)
    test_dataset = get_dataset('report', 'test', False, transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print("Encoding train set...")
    train_img, train_txt, train_lbl = encode_dataset(train_loader)

    print("Encoding val set...")
    val_img, val_txt, val_lbl = encode_dataset(val_loader)

    print("Encoding test set...")
    test_img, test_txt, test_lbl = encode_dataset(test_loader)

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
