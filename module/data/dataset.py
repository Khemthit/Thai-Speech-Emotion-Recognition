import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

def wav2vec2_collate_fn(batch, processor):
    """
    Custom collate function for Wav2Vec2 with padding
    """
    input_values = [item['input_values'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Padding input_values
    padded_inputs = processor.pad(
        {"input_values": input_values},
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )

    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)  # [batch_size]

    return {
        "input_values": padded_inputs["input_values"],
        "attention_mask": padded_inputs["attention_mask"],
        "labels": labels
    }
        
class Wav2VecDataset(Dataset):
    def __init__(self, paths, labels, processor, target_sr=16000, max_duration=6.0):
        self.paths = paths
        self.labels = labels
        self.processor = processor
        self.target_sr = target_sr
        self.max_duration = max_duration

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]

        # Load audio with torchaudio
        audio, orig_sr = torchaudio.load(path)
        
        # Resample if necessary
        if orig_sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=self.target_sr)
            audio = resampler(audio)
        
        # Convert to mono if necessary
        if audio.shape[0] > 1:  # Check if stereo
            audio = audio.mean(dim=0, keepdim=True)

        # Trim or pad the audio to max_duration
        num_samples = int(self.target_sr * self.max_duration)
        if audio.shape[1] > num_samples:
            audio = audio[:, :num_samples]
        elif audio.shape[1] < num_samples:
            padding = num_samples - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, padding))

        # Process the audio with the processor
        inputs = self.processor(
            audio.squeeze(0).numpy(),  # Convert to NumPy for processor compatibility
            sampling_rate=self.target_sr,
            return_tensors="pt"
        )
        
        input_values = inputs["input_values"].squeeze(0)

        return {
            "input_values": input_values,
            "labels": label
        }