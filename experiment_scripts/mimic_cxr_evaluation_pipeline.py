import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from experiment_scripts.evaluation_pipeline import BaseEvaluationPipeline
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from models.vlm_models import LinearProjectionHead
from data.mimic_dataloader import MIMICCXRDataloader, MIMICCXRConfig

class MimicCxrEvaluationPipeline(BaseEvaluationPipeline):
    def __init__(self, args):
        super().__init__(args)
        self._prepare_dataloaders()

    def _prepare_dataloaders(self):
        cfg = MIMICCXRConfig(
            root='/cluster/projects/mcintoshgroup/publicData/MIMIC-CXR/MIMIC-CXR-JPG',
            metadata_csv="/cluster/projects/mcintoshgroup/publicData/fine-grain/MIMIC-CXR-max-metadata/mimic-cxr-2.0.0-metadata.csv.gz",
            split_csv="/cluster/projects/mcintoshgroup/publicData/fine-grain/MIMIC-CXR-max-metadata/mimic-cxr-2.0.0-split.csv.gz",
            image_filenames_txt="/cluster/projects/mcintoshgroup/publicData/fine-grain/MIMIC-CXR-max-metadata/IMAGE_FILENAMES",
            label_col=None,
            chexpert_csv='/cluster/projects/mcintoshgroup/publicData/fine-grain/MIMIC-CXR-max-metadata/mimic-cxr-2.0.0-chexpert.csv.gz',
            transform=self.transform,
            target_transform=None,
            mask_uncertain_labels=self.mask_uncertain_labels,
            override_master_csv=False,
            caption_max_len=self.max_text_len #256 128
        )

        train_dataset = MIMICCXRDataloader(cfg, self.tokenizer, "train")
        val_dataset = MIMICCXRDataloader(cfg, self.tokenizer, "val")
        test_dataset = MIMICCXRDataloader(cfg, self.tokenizer, "test")
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        self.label_classes = train_dataset.classes

        print("Classes:", train_dataset.classes)
        print("Number of training samples:", len(train_dataset))
        print("Number of validation samples:", len(val_dataset))
        print("Number of testing samples:", len(test_dataset))

    def fine_tune_classifier_and_evaluate(self, modality='image'):
        assert modality in ['image', 'text']

        # define the feature tensor as training and validation data
        print('Extracting features')
        train_feats, train_labels = self._extract_image_feats_labels(self.train_data) if modality == 'image' else self._extract_text_feats_labels(self.train_data)
        val_feats, val_labels = self._extract_image_feats_labels(self.val_data) if modality == 'image' else self._extract_text_feats_labels(self.val_data)
        test_feats, test_labels = self._extract_image_feats_labels(self.test_data) if modality == 'image' else self._extract_text_feats_labels(self.test_data)

        # define classifier
        classifier = LinearProjectionHead(train_feats.shape[1], train_labels.shape[-1]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(reduction='mean' if not self.mask_uncertain_labels else 'none')
        optimizer = torch.optim.Adam(classifier.parameters(), lr=self.lr)

        # NOTE: the following should be universal to few-shot, fine-tuning, and custom linear classifier evaluation
        print("Training classifier...")
        self.train_classifier(
            train_feats, 
            train_labels, 
            val_feats, 
            val_labels,
            classifier, 
            criterion, 
            optimizer, 
            self.device,
            mask_uncertain_labels=self.mask_uncertain_labels,
            patience=self.patience, 
            max_epochs=self.epochs,
            batch_size=self.batch_size)

        metrics = self._evaluate_classifier(classifier, test_feats, test_labels)
        out_path = f"/cluster/projects/mcintoshgroup/publicData/fine-grain/experiment/fine_tune_mimic/{self.experiment_model}_results.csv"
        self._save_results(metrics, out_path)

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

    def retrieval(self, topk):
        """use the test split for retrieval"""
        assert 1 <= topk and topk <= 100
        img_feats, txt_feats, _ = self._extract_paired_image_text_features_labels(self.val_data)
        self.i2t_t2i(txt_feats, img_feats, topk)

    def zero_shot_evaluation(self, dataloader):
        """use the test split for retrieval"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_predictions, all_gt_labels = [], []        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                # check preprocess_data.py
                tensor_images = batch['image']
                labels = batch['target']

                tensor_images = tensor_images.to(device)
                labels = labels.to(device)

                # compute normalized image embeddings
                img_feats = self.image_encoder(tensor_images)
                img_feats = self._l2norm(self.image_projection(img_feats))

                # perform multi-hot label classification by performing binary classification on each class
                batch_preds, batch_labels = [], [] # list of multi-hot labels
                for i in range(img_feats.shape[0]):
                    img_feat = img_feats[i]
                    multi_hot_probs = []
                    for class_label in self.label_classes:
                        prompts = ["No " + class_label, class_label]
                        prompt_tokens = self.tokenizer(
                            prompts,
                            return_tensors="pt",
                            truncation=True,
                            padding="longest", # differ than max_len truncation
                            max_length=self.max_text_len
                        )

                        # compute normalized text embeddings
                        prompt_tokens = prompt_tokens.to(device)
                        prompt_feats = self.text_encoder(
                            input_ids=prompt_tokens['input_ids'].squeeze(), 
                            attention_mask=prompt_tokens['attention_mask'].squeeze(), 
                            token_type_ids=prompt_tokens['token_type_ids'].squeeze()
                            )
                        if not isinstance(prompt_feats, torch.Tensor):
                            prompt_feats = prompt_feats.last_hidden_state[:, 0, :]
                        prompt_feats = self._l2norm(self.text_projection(prompt_feats))

                        # compute pairwise cosine similarity and softmax between each of the two prompt with the image embedding (img_feat) and 
                        # find the prompt with the highest cosine similarity, and append the corresponding prompt index to multi_hot_label
                        sim = torch.matmul(img_feat, prompt_feats.T)  # [2]
                        prob = torch.softmax(sim, dim=-1)[1].item()  # confidence of positive class
                        multi_hot_probs.append(prob)

                    batch_preds.append(multi_hot_probs)
                    batch_labels.append(labels[i].cpu().tolist())  # ensure it's on CPU for sklearn

                # list of list NXC, N is number of instances and C is number of classes
                all_predictions.extend(batch_preds)
                all_gt_labels.extend(batch_labels)

        # Convert to NumPy arrays
        all_predictions = np.array(all_predictions).astype(float)  # NxC, float
        all_gt_labels = np.array(all_gt_labels).astype(int)      # NxC, int
        all_gt_labels[all_gt_labels == -1] = 0 # make all the uncertain labels as 0 NOTE:

        # Accuracy using thresholded predictions
        binary_preds = (all_predictions >= 0.5).astype(int)
        accuracy = (binary_preds == all_gt_labels).mean()

        # Compute evaluation metrics
        macro_auc = roc_auc_score(all_gt_labels, all_predictions, average='macro')
        weighted_auc = roc_auc_score(all_gt_labels, all_predictions, average='weighted')
        micro_auc = roc_auc_score(all_gt_labels, all_predictions, average='micro')

        macro_ap = average_precision_score(all_gt_labels, all_predictions, average='macro')
        weighted_ap = average_precision_score(all_gt_labels, all_predictions, average='weighted')
        micro_ap = average_precision_score(all_gt_labels, all_predictions, average='micro')

        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC - macro: {macro_auc:.4f}, weighted: {weighted_auc:.4f}, micro: {micro_auc:.4f}")
        print(f"PR AUC  - macro: {macro_ap:.4f}, weighted: {weighted_ap:.4f}, micro: {micro_ap:.4f}")