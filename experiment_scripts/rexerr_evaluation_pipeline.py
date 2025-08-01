import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.rexerr_dataloader import RexErrDataloader
from experiment_scripts.evaluation_pipeline import BaseEvaluationPipeline
from torch.utils.data import DataLoader
import joblib
from types import SimpleNamespace
import torch
from tqdm import tqdm
from typing_extensions import override

class RexErrEvaluationPipeline(BaseEvaluationPipeline):
    """follow the MimicCxrEvaluationPipeline"""
    def __init__(self, args, is_study_level_sampling, err_level):
        super().__init__(args)
        self.is_study_level_sampling = is_study_level_sampling
        assert self.is_study_level_sampling == False
        self.err_level = err_level
        self._prepare_dataloaders()

        self.encoded_data = None

    def _prepare_dataloaders(self):
        train_dataset = RexErrDataloader(
            f'/cluster/projects/mcintoshgroup/publicData/rexerr-v1/ReXErr-{self.err_level}-level/ReXErr-{self.err_level}-level_train.csv',
            '/cluster/projects/mcintoshgroup/publicData/MIMIC-CXR/MIMIC-CXR-JPG',
            study_level_sampling=self.is_study_level_sampling,
            transform=self.transform,
            tokenizer=self.tokenizer,
            caption_max_len=self.max_text_len,
        )

        val_dataset = RexErrDataloader(
            f'/cluster/projects/mcintoshgroup/publicData/rexerr-v1/ReXErr-{self.err_level}-level/ReXErr-{self.err_level}-level_val.csv',
            '/cluster/projects/mcintoshgroup/publicData/MIMIC-CXR/MIMIC-CXR-JPG',
            study_level_sampling=self.is_study_level_sampling,
            transform=self.transform,
            tokenizer=self.tokenizer,
            caption_max_len=self.max_text_len,
        )

        test_dataset = RexErrDataloader(
            f'/cluster/projects/mcintoshgroup/publicData/rexerr-v1/ReXErr-{self.err_level}-level/ReXErr-{self.err_level}-level_test.csv',
            '/cluster/projects/mcintoshgroup/publicData/MIMIC-CXR/MIMIC-CXR-JPG',
            study_level_sampling=self.is_study_level_sampling,
            transform=self.transform,
            tokenizer=self.tokenizer,
            caption_max_len=self.max_text_len,
        )

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def rexerr_version_retrieval(self, data, topk):
        """check how much disturbance the noise imposes to the final results"""

        img_feats = torch.stack([d.image_feats for d in data], dim=0).to(self.device)   # Shape [N, C]
        paired_txt_feats = torch.stack([d.text_feats for d in data], dim=0).to(self.device)   # Shape [N, C]
        err_txt_feats = torch.stack([d.err_text_feats for d in data], dim=0).to(self.device)  # Shape [M, C]
        txt_feats = torch.cat([paired_txt_feats, err_txt_feats], dim=0).to(self.device)     # Shape [N+M, C]
        num_paired = paired_txt_feats.size(0) # indicate the ground truths
        
        # TEXT-TO-IMAGE (T2I) RETRIEVAL
        # Only evaluate T2I for the first N text features (paired ones), others are invalid computation.
        paired_txt_feats = txt_feats[:num_paired]
        sim_t2i = paired_txt_feats @ img_feats.T  # [N, N]
        gt_indices = torch.arange(num_paired, device=self.device)  # Ground truth

        _, topk_indices_t2i = sim_t2i.topk(k=topk, dim=1)
        hits_t2i = (topk_indices_t2i == gt_indices.unsqueeze(1)).any(dim=1).float()
        recall_t2i = hits_t2i.mean().item()

        # --- IMAGE-TO-TEXT (I2T) with actual error embeddings ---
        sim_i2t_err = img_feats @ txt_feats.T  # [N, N+M]
        _, topk_indices_i2t_err = sim_i2t_err.topk(k=topk, dim=1)
        hits_i2t_err = (topk_indices_i2t_err == gt_indices.unsqueeze(1)).any(dim=1).float()
        recall_i2t_err = hits_i2t_err.mean().item()

        results = {
            f"Recall@{topk}_T2I": recall_t2i,
            f"Recall@{topk}_I2T_err": recall_i2t_err,
        }

        print(f"Retrieval performance with distractors for top-{topk}:")
        print(f"  T2I       -> {recall_t2i:.4f}")
        print(f"  I2T_err   -> {recall_i2t_err:.4f}")
        return results

    def retrieval(self, topk):
        """original retrieval final results"""
        img_feats = self._extract_image_feats_labels(self.test_data, return_label=False)
        txt_feats = self._extract_text_feats_labels(self.test_data, return_label=False)
        self.i2t_t2i(txt_feats, img_feats, topk)

    def zero_shot_evaluation(self):
        pass

    def fine_tune_classifier_and_evaluate(self):
        pass

    @override
    def encode_dataset(self, dataloader, pickle_dest, device):
        """override the original method"""        
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
                # labels = batch['target']
                token_caption = batch['caption']
                token_err_caption = batch['error_caption']

                if len(tensor_images) == 1:
                    tensor_images = tensor_images[0]
                else:
                    print('Current not support study level sampling')
                    assert False
                tensor_images = tensor_images.to(device)
                token_caption = token_caption.to(device)
                token_err_caption = token_err_caption.to(device)

                # Encode and project the image features
                img_feats = self.image_encoder(tensor_images)  # shape: [N, D]
                img_feats = self.image_projection(img_feats)    # apply projector: shape: [N, D']

                # Encode and project caption features
                caption_feat = self.text_encoder(
                    input_ids=token_caption['input_ids'].squeeze(), 
                    attention_mask=token_caption['attention_mask'].squeeze(), 
                    token_type_ids=token_caption['token_type_ids'].squeeze()
                    )
                if not isinstance(caption_feat, torch.Tensor):
                    caption_feat = caption_feat.last_hidden_state[:, 0, :]
                caption_feats = self.text_projection(caption_feat)

                # Encode and project error caption features
                err_caption_feats = self.text_encoder(
                    input_ids=token_err_caption['input_ids'].squeeze(), 
                    attention_mask=token_err_caption['attention_mask'].squeeze(), 
                    token_type_ids=token_err_caption['token_type_ids'].squeeze()
                    )
                if not isinstance(err_caption_feats, torch.Tensor):
                    err_caption_feats = err_caption_feats.last_hidden_state[:, 0, :]
                err_caption_feats = self.text_projection(err_caption_feats)

                # normalize the feature vectors
                # NOTE: after encoding the text and image features, we can do retrieval, few-shot and fine-tune and etc.
                img_feats = self._l2norm(img_feats)
                caption_feats = self._l2norm(caption_feats)
                err_caption_feats = self._l2norm(err_caption_feats)

                # individual feature/study as a dictionary in the dict
                for i, _id in enumerate(study_id):
                    encoded_data.append(SimpleNamespace(**{
                        'study_id': _id,
                        'text_feats': caption_feats[i].cpu(),
                        'err_text_feats': err_caption_feats[i].cpu(),
                        'image_feats': img_feats[i].cpu(),
                    }))
                
                # #NOTE: temporary for testing
                # if len(encoded_data) >= 256:
                #     return encoded_data

        self.encoded_data = encoded_data
        os.makedirs(os.path.dirname(pickle_dest), exist_ok=True)
        joblib.dump(encoded_data, pickle_dest, compress=3)

        return self.encoded_data