from abc import ABCMeta, abstractmethod
import copy
import os.path as osp
import json

from torch.utils.data import Dataset

from ..pipeline import Compose


class BaseDataset(Dataset, metaclass=ABCMeta):

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super().__init__()

        self.ann_file = ann_file
        if data_prefix is not None:
            self.data_prefix = osp.realpath(data_prefix) if osp.isdir(
                data_prefix) else data_prefix
        self.test_mode = test_mode
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.data_infos = self.load_annotations()
        
    @abstractmethod
    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""
        pass

    @abstractmethod
    def evaluate(self, results, metrics):
        """Evaluation for the dataset.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
            logger (logging.Logger | None): Logger for recording.

        Returns:
            dict: Evaluation results dict.
        """
        pass

    def dump_results(self, results, out=None):
        """Dump data to json/yaml/pickle strings or files."""
        if out is None:
            return json.dumps(results)
        elif isinstance(out, str):
            with open(out, 'w') as f:
                json.dump(results, f)
        else:
            raise TypeError('"file" must be a filename str.')

    def prepare_train_data(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.data_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)