

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiment_scripts.rexerr_evaluation_pipeline import RexErrEvaluationPipeline
from experiment_scripts.mimic_cxr_evaluation_pipeline import MimicCxrEvaluationPipeline
import argparse

def str2bool(v):
    return v.lower() in ('true', '1', 'yes')
def parse_args():
    parser = argparse.ArgumentParser(description="Training script with configurable parameters")
    # mgca_resnet_50.ckpt r50_m.tar
    parser.add_argument("--model", type=str, default="r50_m.tar", help="pretrained model checkpoint file name")
    parser.add_argument("--max_text_len", type=int, default=128, help="128 for mgca, 256 for cxrclip")

    # non-relevant to the retrieval task.
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-2, help="Learning rate")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience") # 100
    parser.add_argument("--epochs", type=int, default=800, help="Number of training epochs") # 800
    parser.add_argument("--prediction_threshold", type=float, default=0.5, help="Threshold for binary classification")
    parser.add_argument("--encode_data_only", type=str2bool, default=False, help="is encode data only")
    parser.add_argument("--verify_data_path", type=str2bool, default=False, help="verify data paths")
    parser.add_argument("--mask_uncertain_labels", type=str2bool, default=True, help="mask chestpert labels (-1)")
    parser.add_argument("--fine_tune_modal", type=str, default='text', help="image or text")
    parser.add_argument("--eval_dataset", type=str, default='rexerr', help="in ['mimic', 'rexerr']")

    # rexerr related
    parser.add_argument("--is_study_level_sampling", type=bool, default=False, help="study level sampling")
    parser.add_argument("--error_level", type=str, default='report', help="in ['report', 'sentence']")

    return parser.parse_args()

def main():
    args = parse_args()
    assert args.eval_dataset in ['mimic', 'rexerr']
    print(f'Evaluation dataset: {args.eval_dataset}')
    
    if args.eval_dataset == 'mimic':
        pipeline = MimicCxrEvaluationPipeline(args)
    elif args.eval_dataset == 'rexerr':
        pipeline = RexErrEvaluationPipeline(args, is_study_level_sampling=args.is_study_level_sampling, err_level=args.error_level)

    pipeline.encode_splits(train=False, val=False, test=True, pickle_dest='/cluster/projects/mcintoshgroup/publicData/fine-grain/cache/fine_tune_rexerr/')

    # rexerr retrieval
    pipeline.rexerr_version_retrieval(pipeline.test_data, topk=1)
    pipeline.rexerr_version_retrieval(pipeline.test_data, topk=5)
    pipeline.rexerr_version_retrieval(pipeline.test_data, topk=10)
    pipeline.rexerr_version_retrieval(pipeline.test_data, topk=50)

    # ordinary retrieval
    pipeline.retrieval(topk=1)
    pipeline.retrieval(topk=5)
    pipeline.retrieval(topk=10)
    pipeline.retrieval(topk=50)

if __name__ == '__main__':
    main()