import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Union, Literal
from sklearn.preprocessing import LabelEncoder

class NormalizeSample(object):
    def __init__(self,
                 center_feats: bool = True,
                 scale_feats: bool = True):
        self.center_feats = center_feats
        self.scale_feats = scale_feats

    def __call__(self, sample):
        # sample shape: [time_dim, feature_dim]
        if self.center_feats:
            sample = sample - torch.unsqueeze(sample.mean(dim=-1), dim=-1)
        if self.scale_feats:
            sample = sample / torch.sqrt(torch.unsqueeze(sample.var(dim=-1), dim=-1) + 1e-8)
        return sample


class SliceDataset(Dataset):
    def __init__(self, 
                 data, 
                 labels, 
                 data_subset: Literal['train', 'val'],
                 encoder_engine: Union[LabelEncoder],
                 transform: Optional[torchaudio.transforms.MelSpectrogram] = None,
                 center_feats: bool = True,
                 scale_feats: bool = True,
                 max_len: int = 5,
                 len_thresh: float = 0.5,
                 pad_fn: Literal['pad_zero', 'pad_dup'] = 'pad_zero'
                 ):
        assert isinstance(max_len, int)
        assert data_subset in ['train', 'val'], "data_subset must be either 'train' or 'val'"       
        self.diary = data
        self.transform = transform
        self.max_len = max_len
        self.len_thresh = len_thresh
        self.pad_fn = getattr(self, pad_fn)
        self.normalize = NormalizeSample(center_feats, scale_feats)

        # Fit Encoder on labels
        self.encoder = encoder_engine
        if data_subset == 'train':
            transformed_labels = self.encoder.fit_transform(labels)
        else:
            transformed_labels = self.encoder.transform(labels)

        if isinstance(self.encoder, LabelEncoder):
            self.labels_encoder = transformed_labels
        else:
            raise ValueError("Unsupported encoder type provided.")

        self.samples = self._data_processing()
        print(f"Number of samples in {data_subset} dataset: {len(self.samples)}")
    
    # Pad function
    def pad_dup(self, x: torch.Tensor, max_len: int) -> torch.Tensor:
        """Pad a feature tensor up to the specified time_frames length.
        The tensor is repeated until max_len is reached.
        """
        current_len = x.shape[0]  # time_frames
        if current_len >= max_len:
            return x[:max_len, :]
        else:
            num_repeat = max_len // current_len
            remainder = max_len % current_len
            x_padded = x.repeat(num_repeat, 1)
            if remainder > 0:
                x_padded = torch.cat([x_padded, x[:remainder, :]], dim=0) # Concatenate the remainder
            return x_padded
        
    # Pad function   
    def pad_zero(self, x: torch.Tensor, max_len: int) -> torch.Tensor:
        """Pad Arguments feature up to specified length.
        The padded values are zero.
        Arguments
        """
        current_len = x.shape[0]  # time_dim
        if current_len >= max_len:
            return x[:max_len, :]
        zeros = torch.zeros((max_len - current_len, x.shape[1]), dtype=x.dtype, device=x.device)
        # Concatenate zeros to the input tensor
        x_padded = torch.cat([x, zeros], dim=0)
        return x_padded

    def _load_feature(self, audio_path: str) -> torch.Tensor:
        audio, _ = torchaudio.load(audio_path)
        # Audio has multiple channels, convert to `mono` by averaging the channels
        return torch.unsqueeze(audio.mean(dim=0), dim=0) if audio.shape[0] > 1 else audio
    
    def _chop_sample(self, sample: torch.Tensor) -> List[torch.Tensor]:
        """Chop spectrogram (or any 2-D Tensor) into smaller chunks based on `self.max_len`.
        Each chunk is normalized according to object's config (center_feats, scale_feats).

        :param sample: Input tensor [time_dim, feature_dim]. Can be either spectrogram.
        :return: a list of chopped samples
        """
        x, y = sample["feature"], sample["label"]
        time_dim, _ = x.shape  # time_dim now corresponds to shape[0]
        x_chopped = list()

        # Chop data into chunks of size `self.max_len`
        for i in range(time_dim):
            if i % self.max_len == 0 and i != 0:  # if reach `self.max_len`
                xi = x[i - self.max_len:i, :]  # Adjust slicing to operate on time_dim (shape[0])
                assert xi.shape[0] == self.max_len, xi.shape  # Check the chunk size
                x_chopped.append({"feature": self.normalize(xi), "label": y})

        # If total time_dim is less than `self.max_len`
        if time_dim < self.max_len:
            if self.pad_fn:
                xi = self.pad_fn(x, max_len=self.max_len)
                assert xi.shape[0] == self.max_len
            else:
                xi = x
            x_chopped.append({"feature": self.normalize(xi), "label": y})
        else:  # If file is longer than `self.max_len`, handle remainder
            remainder = x[x.shape[0] - x.shape[0] % self.max_len:, :]  # Slice based on time_dim
            if not remainder.shape[0] <= self.len_thresh:
                if self.pad_fn:
                    xi = self.pad_fn(remainder, max_len=self.max_len)
                else:
                    xi = remainder
                x_chopped.append({"feature": self.normalize(xi), "label": y})
                
        return x_chopped
    
    def _data_processing(self) -> List[Tuple[List[torch.Tensor], torch.Tensor]]:
        samples = []
        for idx, audio_path in tqdm(enumerate(self.diary), desc="Processing", total=len(self.diary)):
            feature = self._load_feature(audio_path)
            if self.transform is not None:
                feature = self.transform(feature).squeeze(0)
            # Transpose so feature shape is [time_dim, feature_dim]
            feature = torch.transpose(feature, 0, 1).to(torch.float32)
            label = torch.tensor(self.labels_encoder[idx], dtype=torch.int)
            samples += (self._chop_sample({"feature": feature, "label":  label}))
            
        return samples

    def __len__(self):
        return len(self.diary)

    def __getitem__(self, idx):
        return self.samples[idx]