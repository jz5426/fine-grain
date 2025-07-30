import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiment_scripts.evaluation_pipeline import BaseEvaluationPipeline
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.vlm_models import LinearProjectionHead
from data.preprocess_mimic_cxr_jpg import MIMICCXRDataloader, MIMICCXRConfig

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
    
    def retrieval(self, topk):
        """use the test split for retrieval"""
        assert 1 <= topk and topk <= 100

        # TODO: DOUBLE CHECK THE FOLLOWING IMPLEMENTATION, SOMETHING NOT RIGHT
        # img_feats and txt are of the same shape [x, y], x is number of feature embeddings, y is the dimension of the features
        # i want to perform text-to-image retrieval for each image feature in the img_feats tensor and also do the same for image-to-text retrieval
        # that is, for each image embedding, whether the corresponding paired text_embedding appears in the topk most similar text embeddings
        # at the end, i want to return the topk retrival recall metrics.
        # recall

        img_feats, txt_feats, _ = self._extract_paired_image_text_features_labels(self.test_data)
        device = img_feats.device

        # Cosine similarity: sim[i, j] = similarity between text i and image j
        sim_matrix = txt_feats @ img_feats.T  # [N_text, N_image]
        num_samples = sim_matrix.size(0)
        gt_indices = torch.arange(num_samples, device=device)

        ### TEXT-TO-IMAGE RETRIEVAL (T2I)
        _, topk_indices_t2i = sim_matrix.topk(k=topk, dim=1, largest=True)
        hits_t2i = (topk_indices_t2i == gt_indices.unsqueeze(1)).any(dim=1).float()
        recall_t2i = hits_t2i.mean().item()

        ### IMAGE-TO-TEXT RETRIEVAL (I2T)
        sim_matrix_i2t = sim_matrix.T  # [N_image, N_text]
        _, topk_indices_i2t = sim_matrix_i2t.topk(k=topk, dim=1, largest=True)
        hits_i2t = (topk_indices_i2t == gt_indices.unsqueeze(1)).any(dim=1).float()
        recall_i2t = hits_i2t.mean().item()

        return {
            f"Recall@{topk}_T2I": recall_t2i,
            f"Recall@{topk}_I2T": recall_i2t
        }

    def zero_shot_evaluation(self):
        """use the test split for retrieval"""

        return 