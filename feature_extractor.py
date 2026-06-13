import numpy as np
import librosa
# from src.config import IMG_SIZE

SR = 8000
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 128
IMG_SIZE=(64,64)
def extract_mel_spectrogram(file_path):
    """
    Convert a WAV file to a normalized Mel spectrogram (64x64x1)
    """
    y, sr = librosa.load(file_path, sr=SR)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                         n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to 0-1
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

    # Pad or trim to IMG_SIZE
    if mel_db.shape[1] < IMG_SIZE[1]:
        pad_width = IMG_SIZE[1] - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0),(0,pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :IMG_SIZE[1]]

    return mel_db[..., np.newaxis]  # shape (64,64,1)