import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiment_scripts.evaluation_pipeline import BaseEvaluationPipeline

class RexErrEvaluationPipeline(BaseEvaluationPipeline):
    def __init__(self, args):
        super().__init__(args)
        self._prepare_dataloaders()

    def _prepare_dataloaders(self):
        pass