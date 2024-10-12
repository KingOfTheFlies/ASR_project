from collections import defaultdict 
from torch.nn.utils.rnn import pad_sequence
import torch

#TODO: change
def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    result_batch_temp = defaultdict(list)

    for dataset_item in dataset_items:
        for key, value in dataset_item.items():
            if type(value) is not str:
                value = value.squeeze(0).T
            result_batch_temp[key].append(value)

    result_batch = defaultdict(list)
    for key, value in result_batch_temp.items():
        if type(value[0]) is not str:
            result_batch[key + '_length'] = torch.tensor([val.shape[0] for val in result_batch_temp[key]])
            result_batch[key] = pad_sequence(value, batch_first=True, padding_value=0)

            if key == 'spectrogram':
                result_batch[key] = result_batch[key].transpose(1, 2)
        else:
            result_batch[key] = result_batch_temp[key]
    return result_batch

# def collate_fn(dataset_items):
#     """
#     Collate and pad fields in the dataset items.
#     Converts individual items into a batch.

#     Args:
#         dataset_items (list[dict]): list of objects from dataset.__getitem__.

#     Returns:
#         result_batch (dict[Tensor]): dict containing batch-version of the tensors.
#     """
#     result_batch = {}
#     for key in dataset_items[0].keys():
#         values = []
#         if isinstance(dataset_items[0][key], str):
#             for item in dataset_items:
#                 values.append(item[key])
#             result_batch[key] = values
#         else:
#             lengths = []
#             for item in dataset_items:
#                 value = item[key].squeeze(0).T
#                 lengths.append(value.shape[0])
#                 values.append(value)
#             result_batch[key + '_length'] = torch.tensor(lengths)
#             padded_values = pad_sequence(values, batch_first=True, padding_value=0)
#             if key == 'spectrogram':
#                 padded_values = padded_values.transpose(1, 2)
#             result_batch[key] = padded_values
#     return result_batch