import dgl.backend as F
import pandas as pd
from pkg.dgllife.csv_dataset import MoleculeCSVDataset


class ADMETDataset(MoleculeCSVDataset):
    def __init__(
            self,
            path='data/hERG.csv',
            smiles_to_graph=None,
            smiles_column='Smiles',
            task_names='Label',
            node_featurizer=None,
            edge_featurizer=None,
            load=False,
            log_every=1000,
            cache_file_path=None,
            n_jobs=1,
            error_log=None,
    ):
        data_path = path
        if data_path.endswith('.txt'):
            df = pd.read_csv(data_path, sep='\t')
        else:
            df = pd.read_csv(data_path)
        df = df.dropna(axis=0)
        self.id = df.index

        super(ADMETDataset, self).__init__(
            df, smiles_to_graph,
            node_featurizer,
            edge_featurizer,
            smiles_column=smiles_column,
            task_names=task_names,
            cache_file_path=cache_file_path,
            load=load, log_every=log_every,
            n_jobs=n_jobs, error_log=error_log,
        )

        self._weight_balancing()

        self.labels = self.labels.reshape([-1, 1])

    def _weight_balancing(self):
        """Perform re-balancing for each task.
        It's quite common that the number of positive samples and the
        number of negative samples are significantly different. To compensate
        for the class imbalance issue, we can weight each datapoint in
        loss computation.
        In particular, for each task we will set the weight of negative samples
        to be 1 and the weight of positive samples to be the number of negative
        samples divided by the number of positive samples.
        If weight balancing is performed, one attribute will be affected:
        * self._task_pos_weights is set, which is a list of positive sample weights
          for each task.
        """
        num_pos = F.sum(self.labels, dim=0)
        num_indices = F.sum(self.mask, dim=0)
        self._task_pos_weights = (num_indices - num_pos) / num_pos

    @property
    def task_pos_weights(self):
        """Get weights for positive samples on each task
        It's quite common that the number of positive samples and the
        number of negative samples are significantly different. To compensate
        for the class imbalance issue, we can weight each datapoint in
        loss computation.
        In particular, for each task we will set the weight of negative samples
        to be 1 and the weight of positive samples to be the number of negative
        samples divided by the number of positive samples.
        Returns
        -------
        Tensor of dtype float32 and shape (T)
            Weight of positive samples on all tasks
        """
        return self._task_pos_weights

    def set_mean_and_std(self, mean=None, std=None):
        """Set mean and std or compute from labels for future normalization.
        The mean and std can be fetched later with ``self.mean`` and ``self.std``.
        Parameters
        ----------
        mean : float32 tensor of shape (T)
            Mean of labels for all tasks.
        std : float32 tensor of shape (T)
            Std of labels for all tasks.
        """
        import numpy as np
        labels = np.array([i.numpy() for i in self.labels])
        if mean is None:
            mean = np.mean(labels, axis=0)
        if std is None:
            std = np.std(labels, axis=0)
        self.mean = mean
        self.std = std
