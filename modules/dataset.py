from torch.utils.data import Dataset


class MultiTaskDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': {
                'main': self.encodings['main'][idx],
                'aux1': self.encodings['aux1'][idx],
                'aux2': self.encodings['aux2'][idx],
                'aux3': self.encodings['aux3'][idx],
            }
        }
