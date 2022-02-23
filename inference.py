import numpy as np
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
import torch
from torch.utils.data import DataLoader
from model import Model
import librosa

def infer_signal(model, signal):
    data_layer.set_signal(signal)
    batch = next(iter(data_loader))
    audio_signal, audio_signal_len = batch
    audio_signal, audio_signal_len = audio_signal.to(model.device), audio_signal_len.to(model.device)
    logits = model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
    return logits

class AudioDataLayer(IterableDataset):
    @property
    def output_types(self):
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return torch.as_tensor(self.signal, dtype=torch.float32), \
               torch.as_tensor(self.signal_shape, dtype=torch.int64)

    def set_signal(self, signal):
        self.signal = signal.astype(np.float32)
        self.signal_shape = self.signal.size
        self.output = True

    def __len__(self):
        return 1

if __name__ == "__main__":
    model_path = r"C:\MIE CARTELLE\PROGRAMMAZIONE\GITHUB\tesi_magistrale\nemo_experiments\MatchboxNet-3x2x64\2022-01-19_23-29-46\checkpoints_backup\matchcboxnet--val_loss=0.369-epoch=249.model"
    exp_dir = r"C:\MIE CARTELLE\PROGRAMMAZIONE\GITHUB\tesi_magistrale\nemo_experiments\MatchboxNet-3x2x64\2022-01-19_23-29-46"
    model = Model.load_backup(model_path, exp_dir)
    model = model.eval()
    data_layer = AudioDataLayer(sample_rate=16000)
    data_loader = DataLoader(data_layer, batch_size=1, collate_fn=data_layer.collate_fn)

    audio_path = r"C:\MIE CARTELLE\PROGRAMMAZIONE\GITHUB\tesi_magistrale\nemo_experiments\MatchboxNet-3x2x64\2022-01-19_23-29-46\db\clips\30\2126502572_eng_20.wav"
    signal, sr = librosa.load(audio_path, sr=16000)
    logits: torch.Tensor = infer_signal(model, signal)
    predict_labels = model.predict(logits).cpu().detach().numpy()
    predict_labels = np.argmax(predict_labels, axis=1)
    print(predict_labels)