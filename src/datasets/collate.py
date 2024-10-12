from collections import defaultdict 
from torch.nn.utils.rnn import pad_sequence
import torch

def collate_fn(dataset_items):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from dataset.__getitem__.

    Returns:
        result_batch (dict[Tensor]): dict containing batch-version of the tensors.
    """
    result_batch = {}
    for key in dataset_items[0].keys():
        values = []
        if isinstance(dataset_items[0][key], str):
            for item in dataset_items:
                values.append(item[key])
            result_batch[key] = values
        else:
            lengths = []
            for item in dataset_items:
                value = item[key].squeeze(0).T
                lengths.append(value.shape[0])
                values.append(value)
            result_batch[key + '_length'] = torch.tensor(lengths)
            padded_values = pad_sequence(values, batch_first=True, padding_value=0)
            if key == 'spectrogram':
                padded_values = padded_values.transpose(1, 2)
            result_batch[key] = padded_values
    return result_batch