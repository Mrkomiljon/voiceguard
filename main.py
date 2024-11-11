import argparse
import os
import numpy as np
import librosa
import yaml
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from model import RawNet
import torchaudio
from tqdm import tqdm

SAMPLE_RATE = 24000

class Dataset_LibriSeVoc(Dataset):
    def __init__(self, dataset_path, split='train'):
        self.dataset_path = dataset_path
        self.split = split
        self.cut = SAMPLE_RATE * 4  # 4 sekundlik audio kesimi

        # Fayllarni yuklaymiz
        self.path_list_train, self.y_list_train = self.load_data('train')
        self.path_list_dev, self.y_list_dev = self.load_data('dev')
        self.path_list_test, self.y_list_test = self.load_data('test')

        print(f'Loaded {len(self.path_list_train)} training, {len(self.path_list_dev)} dev, {len(self.path_list_test)} test samples')

    def load_data(self, split):
        path_list = []
        y_list = []
        for subset_name in os.listdir(self.dataset_path):
            subset_path = os.path.join(self.dataset_path, subset_name)
            label = 0 if subset_name.startswith('gt') else 1
            for file_name in os.listdir(subset_path):
                path_list.append(os.path.join(subset_path, file_name))
                y_list.append(label)
        return path_list, y_list

    def __len__(self):
        if self.split == 'train':
            return len(self.path_list_train)
        elif self.split == 'dev':
            return len(self.path_list_dev)
        else:
            return len(self.path_list_test)

    def extract_features(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)

        # 1. MFCC
        mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=40)(waveform)
        mfcc = mfcc.mean(dim=2)

        # 2. Mel-Spectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=512, hop_length=160, n_mels=64)(waveform)
        mel_spec = mel_spec.mean(dim=2)

        # 3. Zero-Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(waveform.numpy().squeeze())[0].mean()

        # 4. Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform.numpy().squeeze(), sr=sample_rate)[0].mean()

        # 5. RMS Energy
        rms = torchaudio.transforms.AmplitudeToDB()(waveform).mean().item()

        # 6. Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=waveform.numpy().squeeze(), sr=sample_rate)[0].mean()

        # Xususiyatlarni birlashtirish
        features = torch.tensor(np.concatenate([
            mfcc.numpy().flatten(),
            mel_spec.numpy().flatten(),
            [zcr, spectral_bandwidth, rms, spectral_contrast]
        ]), dtype=torch.float32)

        return features

    def __getitem__(self, index):
        if self.split == 'train':
            path = self.path_list_train[index]
            label = self.y_list_train[index]
        elif self.split == 'dev':
            path = self.path_list_dev[index]
            label = self.y_list_dev[index]
        else:
            path = self.path_list_test[index]
            label = self.y_list_test[index]

        features = self.extract_features(path)
        return features, label


def evaluate_accuracy(dev_loader, model, device):
    num_correct = 0
    num_total = 0
    model.eval()
    
    for batch_features, batch_labels in dev_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        outputs = model(batch_features)
        
        _, preds = torch.max(outputs, 1)
        num_correct += (preds == batch_labels).sum().item()
        num_total += batch_labels.size(0)
    
    accuracy = (num_correct / num_total) * 100
    return accuracy


def train_epoch(train_loader, model, optimizer, device, criterion):
    model.train()
    running_loss = 0.0
    num_correct = 0
    num_total = 0

    for batch_features, batch_labels in tqdm(train_loader, total=len(train_loader)):
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_features.size(0)
        
        _, preds = torch.max(outputs, 1)
        num_correct += (preds == batch_labels).sum().item()
        num_total += batch_labels.size(0)
    
    accuracy = (num_correct / num_total) * 100
    avg_loss = running_loss / num_total
    return avg_loss, accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    train_set = Dataset_LibriSeVoc(dataset_path=args.data_path, split='train')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    dev_set = Dataset_LibriSeVoc(dataset_path=args.data_path, split='dev')
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False)

    # Load model config
    with open('model_config_RawNet.yaml', 'r') as f_yaml:
        parser1 = yaml.safe_load(f_yaml)

    model = RawNet(parser1['model'], device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(args.num_epochs):
        train_loss, train_acc = train_epoch(train_loader, model, optimizer, device, criterion)
        valid_acc = evaluate_accuracy(dev_loader, model, device)

        print(f'Epoch {epoch+1}/{args.num_epochs} - Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Dev Acc: {valid_acc:.2f}%')

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(args.model_save_path, f'best_model.pth'))
            print(f'Best model saved with accuracy: {best_acc:.2f}%')
