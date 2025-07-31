import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from models.vlm_models import cxrclip_model, mgca_model
from types import SimpleNamespace
import pandas as pd
from datetime import datetime
import joblib
from abc import ABC, abstractmethod


class BaseEvaluationPipeline(ABC):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_params()
        self._setup_model_and_transform()

    def _init_params(self):
        self.batch_size = self.args.batch_size
        self.lr = self.args.learning_rate
        self.patience = self.args.patience
        self.epochs = self.args.epochs
        self.pred_thresh = self.args.prediction_threshold
        self.model_ckpt = self.args.model
        self.mask_uncertain_labels = self.args.mask_uncertain_labels
        self.max_text_len = self.args.max_text_len
        self.fine_tune_modal = self.args.fine_tune_modal

        # print each argument
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.lr}")
        print(f"Patience: {self.patience}")
        print(f"Epochs: {self.epochs}")
        print(f"Prediction threshold: {self.pred_thresh}")
        print(f"Model checkpoint: {self.model_ckpt}")
        print(f"Mask uncertain labels: {self.mask_uncertain_labels}")
        print(f"Max text length: {self.max_text_len}")
        print(f"Fine-tune modality: {self.fine_tune_modal}")

    def _setup_model_and_transform(self):
        if self.model_ckpt in ['r50_mcc.tar', 'r50_mc.tar', 'r50_m.tar']:
            self.model_name = 'cxrclip'
            self.input_size = 224
            self.model_specific_cache_dir = 'cxrclip_encoder_features'

            self.experiment_model = f"cxrclip_{self.model_ckpt.split('.')[0]}"
            self.vlm = cxrclip_model(
                f'/cluster/projects/mcintoshgroup/publicData/fine-grain/CXR-CLIP-Image-Encoder/{self.model_ckpt}', 
                '/cluster/projects/mcintoshgroup/publicData/fine-grain/CXR-CLIP-Text-Encoder/', 
                '/cluster/projects/mcintoshgroup/publicData/fine-grain/CXR-CLIP-Text-Encoder/'
                )

            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            self.transform = Compose([
                Resize((self.input_size, self.input_size)),
                CenterCrop((self.input_size, self.input_size)),
                ToTensor(),
                Normalize(mean=mean, std=std)
            ])
        elif self.model_ckpt == 'mgca_resnet_50.ckpt':
            self.model_name = 'mgca'
            self.input_size = 224
            self.model_specific_cache_dir = 'mgca_encoder_features'
            self.experiment_model = 'mgca_res50'
            self.vlm = mgca_model(
                f'/cluster/projects/mcintoshgroup/publicData/fine-grain/MGCA-Image-Encoder/{self.model_ckpt}', 
                '/cluster/projects/mcintoshgroup/publicData/fine-grain/CXR-CLIP-Text-Encoder/', 
                '/cluster/projects/mcintoshgroup/publicData/fine-grain/CXR-CLIP-Text-Encoder/'
            )

            mean, std = [0.5]*3, [0.5]*3
            self.transform = Compose([            
                CenterCrop((self.input_size, self.input_size)),
                ToTensor(),
                Normalize(mean=mean, std=std)
            ])
        else:
            raise ValueError(f"Unsupported model checkpoint: {self.model_ckpt}")

        self.tokenizer = self.vlm.tokenizer
        self.image_encoder = self.vlm.image_encoder
        self.text_encoder = self.vlm.text_encoder
        self.image_projection = self.vlm.image_projection
        self.text_projection = self.vlm.text_projection

        for model in [self.image_encoder, self.image_projection, self.text_encoder, self.text_projection]:
            model.to(self.device).eval()

    def encode_splits(self, train=True, val=True, test=True, pickle_dest='/cluster/projects/mcintoshgroup/publicData/fine-grain/cache/fine_tune_mimic/'):
        assert any([train, val, test]) == True

        if train:
            print("Encoding train set...")
            self.train_data = self.encode_dataset(
                self.train_loader, 
                pickle_dest=os.path.join(pickle_dest, f'{self.model_specific_cache_dir}/train_features.joblib'),
                device=self.device)

        if val:
            print("Encoding validation set...")
            self.val_data = self.encode_dataset(
                self.val_loader, 
                pickle_dest=os.path.join(pickle_dest, f'{self.model_specific_cache_dir}/val_features.joblib'),
                device=self.device)

        if test:
            print("Encoding test set...")
            self.test_data = self.encode_dataset(
                self.test_loader, 
                pickle_dest=os.path.join(pickle_dest, f'{self.model_specific_cache_dir}/test_features.joblib'),
                device=self.device)

    @abstractmethod
    def fine_tune_classifier_and_evaluate(self):
        pass

    @abstractmethod
    def retrieval(self, topk):
        pass

    @abstractmethod
    def zero_shot_evaluation(self):
        pass

    # def few_shot_evaluation(self):
    #     """this is potentially optional, not every dataset can have few-shot evaluation such as mimic"""
    #     pass

    # def _extract_paired_image_text_feats_labels(self, data):
    #     """this is for special case rexerr_eval"""
    #     pass

    def _extract_text_feats_labels(self, data):
        feats = torch.stack([d.text_feats for d in data]).to(self.device)
        labels = torch.stack([d.label for d in data]).to(self.device)
        return feats, labels

    def _extract_image_feats_labels(self, data):
        feats = torch.stack([d.image_feats for d in data]).to(self.device)
        labels = torch.stack([d.label for d in data]).to(self.device)
        return feats, labels
    
    def _extract_paired_image_text_features_labels(self, data):
        img_feats, txt_feats, labels = [], [], []
        for d in data:
            img_feats.append(d.image_feats)
            txt_feats.append(d.text_feats)
            labels.append(d.label)
        img_feats = torch.stack(img_feats).to(self.device)
        txt_feats = torch.stack(txt_feats).to(self.device)
        labels = torch.stack(labels).to(self.device)

        return img_feats, txt_feats, labels


    def _evaluate_classifier(self, classifier, feats, labels):
        classifier.eval()
        all_logits = []
        test_loader = DataLoader(TensorDataset(feats, labels), batch_size=self.batch_size)

        with torch.no_grad():
            for x, y in test_loader:
                logits = classifier(x.to(self.device))
                all_logits.append(logits.cpu())

        logits = torch.cat(all_logits)
        probs = torch.sigmoid(logits)
        labels = labels.cpu()
        mask = labels != -1.0

        flat_labels = labels[mask].cpu().numpy()
        flat_probs = probs[mask].cpu().numpy()
        flat_preds = (probs >= self.pred_thresh)[mask].cpu().numpy()

        accuracy = (flat_preds == flat_labels).mean()
        pr_auc = average_precision_score(flat_labels, flat_probs)
        auc = roc_auc_score(flat_labels, flat_probs, average='micro')

        print(f'accuracy: {accuracy} auc: {auc} pr_auc: {pr_auc}')
        return {"accuracy": accuracy, "pr_auc": pr_auc, "auc": auc}

    def _save_results(self, metrics, out_path):
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "batch_size": self.batch_size,
            "input_size": self.input_size,
            "learning_rate": self.lr,
            "patience": self.patience,
            "epochs": self.epochs,
            "prediction_threshold": self.pred_thresh,
            **metrics
        }

        # out_path = f"/cluster/projects/.../{self.experiment_model}_results.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df = pd.DataFrame([results])
        df.to_csv(out_path, mode='a' if os.path.exists(out_path) else 'w', index=False)
        print(f"Results saved to {out_path}")

    def encode_dataset(self, dataloader, pickle_dest, device):
        
        if os.path.exists(pickle_dest):
            try:
                encoded_data = joblib.load(pickle_dest, mmap_mode='r')
                print("[SUCCESS] Object loaded successfully.")
                return encoded_data
            except Exception as e:
                print("[ERROR] Failed to load object.")

        encoded_data = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                # check preprocess_data.py
                tensor_images = batch['image']
                study_id = batch['study_id'] # key of the results
                labels = batch['target']
                token_caption = batch['caption']

                # TODO: currently only support MGCA model, not CXR

                tensor_images = tensor_images.to(device)
                token_caption = token_caption.to(device)

                # Encode and project the image features
                img_feats = self.image_encoder(tensor_images)  # shape: [N, D]
                img_feats = self.image_projection(img_feats)    # apply projector: shape: [N, D']

                # Encode and project caption features
                report_feat = self.text_encoder(
                    input_ids=token_caption['input_ids'].squeeze(), 
                    attention_mask=token_caption['attention_mask'].squeeze(), 
                    token_type_ids=token_caption['token_type_ids'].squeeze()
                    )
                if not isinstance(report_feat, torch.Tensor):
                    report_feat = report_feat.last_hidden_state[:, 0, :]
                # NOTE: check line 98 of mgca_module in the original repo => report_feat is the global embedding tensor
                caption_feats = self.text_projection(report_feat)

                # normalize the feature vectors
                # NOTE: after encoding the text and image features, we can do retrieval, few-shot and fine-tune and etc.
                img_feats = self._l2norm(img_feats)
                caption_feats = self._l2norm(caption_feats)

                # individual feature/study as a dictionary in the dict
                for i, _id in enumerate(study_id):
                    encoded_data.append(SimpleNamespace(**{
                        'study_id': _id,
                        'text_feats': caption_feats[i].cpu(),
                        'image_feats': img_feats[i].cpu(),
                        'label': labels[i].cpu()
                    }))
                
                # #NOTE: temporary for testing
                # if len(encoded_data) >= 256:
                #     return encoded_data

        # Ensure the parent directories exist
        os.makedirs(os.path.dirname(pickle_dest), exist_ok=True)
        joblib.dump(encoded_data, pickle_dest, compress=3)

        return encoded_data
    
    def train_classifier(self, train_x_tensor, train_y_tensor, val_x_tensor, val_y_tensor, classifier, criterion, optimizer, device, mask_uncertain_labels, patience=5, max_epochs=50, batch_size=32):
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

    def _l2norm(self, t):
        return F.normalize(t, dim = -1)
