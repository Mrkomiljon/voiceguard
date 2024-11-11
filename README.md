# **VoiceGuard: Real-Time Voice Phishing Detection using RawNet**

## 📋 **Overview**
VoiceGuard is an AI-powered system designed to detect voice phishing using a deep learning-based model, **RawNet**. The system utilizes advanced signal processing techniques, attention mechanisms, and GRU layers to accurately classify audio samples as either real or synthesized. This tool can be applied to protect users from malicious voice-based attacks, especially in phone conversations or online communications.

---

## 🛠 **Features**
- **Binary Classification**: Distinguishes between real human voices and synthesized ones (i.e., fake).
- **Multi-Class Classification**: Identifies specific voice synthesis techniques (e.g., WaveGrad, DiffWave, MelGAN).
- **Efficient Feature Extraction**: Utilizes Mel-spectrogram, MFCCs, and other audio features for robust analysis.
- **Mobile Deployment**: Supports model quantization for mobile deployment, reducing model size and inference time.

## **Install dependencies**
```
pip install -r requirements.txt
```
🚀 Training the Model
Step 1: Configure the model
Edit the model_config_RawNet.yaml file to set up your model parameters.
```
python train.py --data_path ./dataset --batch_size 64 --num_epochs 50 --lr 0.0001 --model_save_path ./checkpoints
```
Parameters:

- --data_path: Path to your dataset.
- --batch_size: Batch size for training.
-  --num_epochs: Number of epochs to train.
-  --lr: Learning rate.
-  --model_save_path: Directory to save the trained models.
  
🧪 Evaluation
To evaluate the performance of your trained model, use the eval.py script:
```
python eval.py --input_path path/to/audio.wav --model_path checkpoints/epoch_best_model.pth
```
Sample output

```
Device: cuda
Model loaded: models/best_model.pth
Multi classification result:
gt: 0.8794, wavegrad: 0.0251, diffwave: 0.0132, parallel wave gan: 0.0193, wavernn: 0.0178, wavenet: 0.0214, melgan: 0.0238
Binary classification result: fake: 0.0456, real: 0.9544
```
🤝 Contributing
Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.

How to Contribute
1. Fork the repository.
2. Create a feature branch:

```
git checkout -b feature-branch
```
Commit your changes:

```
git commit -m "Add new feature"
```
Push to your branch:

```
git push origin feature-branch
```
📚 Acknowledgements
* [LibriSeVoc](https://drive.google.com/file/d/1NXF9w0YxzVjIAwGm_9Ku7wfLHVbsT7aG/view) Dataset for providing audio samples.
* PyTorch and Torchaudio for efficient deep learning and audio processing capabilities.
* RawNet architecture, which serves as the backbone for our model.
  
📝 Future Improvements
- Extend the system to handle real-time audio streams.
- Integrate the model into mobile applications for on-device voice phishing detection.
- Improve model accuracy using data augmentation techniques.
