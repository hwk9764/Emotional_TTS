import numpy as np
import os
import tgt
from scipy.io.wavfile import read
import pyworld as pw
import torch
import audio as Audio
from utils import get_alignment, standard_norm, remove_outlier, average_by_duration
import hparams as hp
from jamo import h2j
import codecs
import librosa
import soundfile as sf

from sklearn.preprocessing import StandardScaler

# 데이터
def prepare_align(in_dir, meta):
    with open(os.path.join(in_dir, meta), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            basename, text = parts[0], parts[3]

            basename=basename.replace('.wav','.txt')
            
            with open(os.path.join(in_dir,'wavs', basename),'w') as f1:
                f1.write(text)

def build_from_path(in_dir, out_dir, tg_dir, train_meta, val_meta):
    train, val = list(), list()
    scalers = [StandardScaler(copy=False) for _ in range(3)]	# scalers for mel, f0, energy
    with open(os.path.join(in_dir, train_meta), encoding='utf-8') as f:
        for index, line in enumerate(f):
            parts = line.strip().split('|')
            subfolder, basename = parts[0], parts[1]
            ret = process_utterance(in_dir, out_dir, tg_dir, subfolder, basename, scalers)
            if ret is None:
                continue
            else:
                info, n = ret
                
            train.append(info)

            if index % 1000 == 0:
                print("training Done %d" %index)
            
            
    with open(os.path.join(in_dir, val_meta), encoding='utf-8') as f:
        for index, line in enumerate(f):
            parts = line.strip().split('|')
            subfolder, basename = parts[0], parts[1]
            ret = process_utterance(in_dir, out_dir, tg_dir, subfolder, basename, scalers, is_valid=True)
            if ret is None:
                continue
            else:
                info, n = ret
                
            val.append(info)

            if index % 100 == 0:
                print("validation Done %d" %index)
            
            
    param_list = [np.array([scaler.mean_, scaler.scale_]) for scaler in scalers]
    param_name_list = ['mel_stat.npy', 'f0_stat.npy', 'energy_stat.npy']
    [np.save(os.path.join(out_dir, param_name), param_list[idx]) for idx, param_name in enumerate(param_name_list) if not os.path.isfile(os.path.join(out_dir, param_name))]
    return [r for r in train if r is not None], [r for r in val if r is not None]

def process_utterance(in_dir, out_dir, tg_dir, subfolder, basename, scalers, is_valid=False):
    basename=basename.replace('.wav','')
    #wav_bak_path = os.path.join(in_dir, "wavs_bak", "{}.wav".format(wav_bak_basename))
    wav_path = os.path.join(in_dir, subfolder, '{}.wav'.format(basename))
    #tg_path = os.path.join(in_dir, subfolder, '{}.TextGrid'.format(basename))
    tg_path = os.path.join(tg_dir, subfolder, '{}.TextGrid'.format(basename))
    if not os.path.exists(tg_path):
        return None
    # Get alignments
    textgrid = tgt.io.read_textgrid(tg_path)
    phone, duration, start, end = get_alignment(textgrid.get_tier_by_name('phones'))
    text = '{'+ '}{'.join(phone) + '}' # '{A}{B}{$}{C}', $ represents silent phones
    text = text.replace('{$}', ' ')    # '{A}{B} {C}'
    text = text.replace('}{', ' ')     # '{A B} {C}'
    
    if start >= end:
        print('start is bigger than end')
        return None
    
    # Read and trim wav files
    sr, wav = read(wav_path)
                
    if sr != hp.sampling_rate:
        y, sr = librosa.load(wav_path, sr=sr)
        wav = librosa.resample(y=y, orig_sr=sr, target_sr=hp.sampling_rate)
        sf.write(wav_path, wav.astype(np.float32), hp.sampling_rate)
        sr, wav = read(wav_path)
        
    wav = wav[int(hp.sampling_rate*start):int(hp.sampling_rate*end)]#.astype(np.float32)
    #wav = np.asarray(wav, dtype="float32")
    #wav = wav.astype(np.float32)
    
    # Compute fundamental frequency
    f0, t = pw.dio(wav.astype(np.float64), hp.sampling_rate, frame_period=hp.hop_length/hp.sampling_rate*1000)
    # frame_period가 stft의 frame의 길이가 아니라 frame들 간의 거리를 뜻함. sample hop length개만큼의 간격을 가지고 이를 시간으로 변환한 것
    f0 = pw.stonemask(wav.astype(np.float64), f0, t, hp.sampling_rate)
    f0 = f0[:sum(duration)]
    # Compute mel-scale spectrogram and energy

    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(torch.FloatTensor(wav))
    mel_spectrogram = mel_spectrogram.numpy().astype(np.float32)[:, :sum(duration)]
    energy = energy.numpy().astype(np.float32)[:sum(duration)]

    '''
    f0, energy = remove_outlier(f0), remove_outlier(energy)
    # 0이 대부분을 차지하는 pitch contour나 energy는 값이 있는 부분을 전부 이상치로 간주하고 0으로 고쳐서 결국 pitch가 전혀 없는 contour가 됨.
    => 애초에 non-zero값만 뽑아서 이상치를 처리하기 때문에 괜춘.
    # 감정을 표현하는데 중요한 info인만큼 이상치 처리를 하지 않기로 함. => 미친 놈아
    # StandardScaler는 outlier에 민감하여 같은 분포여도 outlier때문에 다르게 표준화될 수 있음. 그래서 outlier 없애는 것.
    
    이하 무시해도 됨
    f0, energy = average_by_duration(f0, duration), average_by_duration(energy, duration)
    # 비유하자면 아날로그 신호를 quantization해서 디지털로 바꾸는 과정. 일정 구간들의 모든 값을 하나의 값으로 일직선을 그어버림 => 정보의 해상도가 떨어짐.
    # 이런 짓을 왜 하는지 모르겠음
    '''
    f0, energy = remove_outlier(f0), remove_outlier(energy)
    f0, energy = average_by_duration(f0, duration), average_by_duration(energy, duration)
        
    if mel_spectrogram.shape[1] >= hp.max_seq_len:
        return None
    # Save alignment
    ali_filename = '{}-ali-{}.npy'.format(hp.dataset, basename)
    temp_dir = os.path.join(out_dir, 'alignment', subfolder)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    np.save(os.path.join(temp_dir, ali_filename), duration, allow_pickle=False) if not os.path.isfile(os.path.join(temp_dir, ali_filename)) else None

    # Save fundamental prequency
    f0_filename = '{}-f0-{}.npy'.format(hp.dataset, basename)
    temp_dir = os.path.join(out_dir, 'f0', subfolder)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    np.save(os.path.join(temp_dir, f0_filename), f0, allow_pickle=False) if not os.path.isfile(os.path.join(temp_dir, f0_filename)) else None

    # Save energy
    energy_filename = '{}-energy-{}.npy'.format(hp.dataset, basename)
    temp_dir = os.path.join(out_dir, 'energy', subfolder)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    np.save(os.path.join(temp_dir, energy_filename), energy, allow_pickle=False) if not os.path.isfile(os.path.join(temp_dir, energy_filename)) else None

    # Save spectrogram
    mel_filename = '{}-mel-{}.npy'.format(hp.dataset, basename)
    temp_dir = os.path.join(out_dir, 'mel', subfolder)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    np.save(os.path.join(temp_dir, mel_filename), mel_spectrogram.T, allow_pickle=False) if not os.path.isfile(os.path.join(temp_dir, mel_filename)) else None
   
    if not is_valid:    # train set에만 fit해 있어야 모델의 성능이 더 좋다고 함. valid도 하면 data leakage 발생
        mel_scaler, f0_scaler, energy_scaler = scalers

        mel_scaler.partial_fit(mel_spectrogram.T)
    
    
        if sum(f0) == 0:
            print('f0 왜 이럼? ', basename)
        if sum(energy) == 0:
            print('energy 왜 이럼? ',basename)
        f0_scaler.partial_fit(f0[f0!=0].reshape(-1, 1))
        energy_scaler.partial_fit(energy[energy != 0].reshape(-1, 1))

    return '|'.join([basename, text]), mel_spectrogram.shape[1]
