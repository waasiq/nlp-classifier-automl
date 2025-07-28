import nlpaug.augmenter.word as naw
from torch.utils.data import Dataset

eda = naw.SynonymAug(aug_src='wordnet')

class AugDataset(Dataset):
    def __init__(self, df, p=0.5):
        self.df, self.p = df, p
    def __len__(self):  return len(self.df)
    def __getitem__(self, idx):
        text, label = self.df.iloc[idx][['text', 'label']]
        if random.random() < self.p:
            text = eda.augment(text)
        return text, label
