"""
- combine both the rexerr data with the mimic-cxr-jpg data based on the p+subject_id/s+study_id/dicom_id in this order.
    - need to collapse the 10 folders in the MIMI-CXR-JPG folder so that it matches the pattern of "p+subject_id/s+study_id/dicom_id"
    - do that for sentence level and report level
    - after you matches the rexerr data with the mimic-cxr-jpg data, save that as a dictionary
    - create a data class for that
    - each iterator output the follows:
        {
            image_path: ...,
            image_tensor: ...,
            caption: ..., # either report or sentence (most likely sentence should be enough)
            error_caption: ..., # either report or sentence (most likely sentence should be enough)
            error_Sampled: ...
        }
        - note that text is being forward pass during training, so no need to 
"""
import os
import csv
import random
from PIL import Image
from torch.utils.data import Dataset

class RexErrDataloader(Dataset):
    def __init__(self, rexerr_path, mimic_cxr_jpg_path, transform, study_level_sampling=True):
        self.rexerr_path = rexerr_path
        self.mimic_cxr_jpg_path = mimic_cxr_jpg_path
        self.data = self.intersec_rexerr_mimicCxrJpg(self.rexerr_path, self.mimic_cxr_jpg_path)
        self.transform = transform
        self.study_level_sampling = study_level_sampling
        
    def __len__(self):
        return len(self.data)

    def intersec_rexerr_mimicCxrJpg(self, rexerr_path, mimicCxrJpg_path):
        # 1. create an empty list
        result = []

        # 2. open the csv file for the rexerr_path
        with open(rexerr_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            # 3. each row in the csv file contains the dicom_id, study_id, subject_id
            for row in reader:
                dicom_ids = row['dicom_id'].split(',')  # 3. handle multiple dicom_ids
                study_id = row['study_id']
                subject_id = row['subject_id']

                # 4. transform subject_id like 10000032 to p10 using the first two digits
                parent_folder = f"p{str(subject_id)[:2]}" # p10
                patient_folder = f"p{str(subject_id)}" # p10000032

                # 5. transform study_id like 50414267 to s50414267
                study_folder = f"s{study_id}"

                # 6. create image path for each dicom_id
                study_id_paths = []
                for dicom_id in dicom_ids:
                    id_path = os.path.join(mimicCxrJpg_path, parent_folder, patient_folder, study_folder, f"{dicom_id.strip()}.jpg")
                    if os.path.exists(id_path):
                        study_id_paths.append(id_path)
                    else:
                        print(id_path, ' does not exists in MIMIC_CXR_JPG directory!! => NOT SAVING IT')

                # TODO: filter based on the sampled error type
                if len(study_id_paths) > 0:
                    result.append({
                        'study_id': study_id, # str/int
                        'image_paths': study_id_paths, # list all image within the study_id share the same text report
                        'caption': row['original_report'], # str
                        'error_caption': row['error_report'], # str
                        'errors_sampled': row['errors_sampled'] # str
                    })

        # 8. return the list
        return result

    def __getitem__(self, index):
        data = self.data[index]
        
        image_paths = data['image_paths'].copy()
        assert len(image_paths) > 0

        # instance based contrastive learning
        if not self.study_level_sampling:
            image_paths = [random.choice(image_paths)]

        # transform individual image
        tensor_images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')  # Ensure it's 3-channel RGB
            tensor_image = self.transform(image)
            tensor_images.append(tensor_image)

        results = {
            'image_paths': image_paths,
            'tensor_images': tensor_images,
            'caption': data['caption'],
            'error_caption':  data['error_caption'],
            'study_id': data['study_id']
        }
        return results

if __name__ == '__main__':
    pass

