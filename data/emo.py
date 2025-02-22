import numpy as np
import os
import tgt
from scipy.io.wavfile import read
import pyworld as pw    # https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder
import torch
import audio as Audio
from utils import get_alignment, standard_norm, remove_outlier, average_by_duration
import hparams as hp
from jamo import h2j
import codecs

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
    # train 데이터 전처리
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
            
    # validation 데이터 전처리
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
    wav_path = os.path.join(in_dir, subfolder, '{}.wav'.format(basename))
    wav_bak_path = os.path.join(in_dir, "wavs_bak", subfolder, "{}.wav".format(basename))
    tg_path = os.path.join(tg_dir, subfolder, '{}.TextGrid'.format(basename))
    if not os.path.exists(tg_path):
        print(f"'{tg_path}' does not exist")
        return None
    
    # Convert kss data into PCM encoded wavs
    if not os.path.isfile(wav_path):
        # 중괄호 두 개 같은 경로로 해도 되는데 덮어쓴다길래 혹시나 해서 bak 씀
        os.system("ffmpeg -i {} -ac 1 -ar 22050 {}".format(wav_bak_path, wav_path))
        
    # Get alignments
    '''
    duration은 각 phoneme이 (발음이) 얼마나 지속되는지를 나타내고 있음
    [8  9  4  1 11 ...] 이런 duration이 있다고 할 때
    첫 번째 phone은 8 frame동안, 두 번째 frame은 그 후 9 frame동안 지속된다는 뜻
    '''
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
                
    wav = wav[int(hp.sampling_rate*start):int(hp.sampling_rate*end)].astype(np.float32)
    #wav = np.asarray(wav, dtype="float32")
    #wav = wav.astype(np.float32)
    
    # Compute fundamental frequency
    f0, t = pw.dio(wav.astype(np.float64), hp.sampling_rate, frame_period=hp.hop_length/hp.sampling_rate*1000)
    # frame_period가 stft의 frame의 길이가 아니라 frame들 간의 거리를 뜻함. sample hop length개만큼의 간격을 가지고 이를 시간으로 변환한 것
    # pw.dio로 얻은 f0를 refine하여 더 정확한 f0를 얻음. pw.dio로 얻은 f0는 noisy하거나 정확하지 않은 경우가 있다고 함.
    f0 = pw.stonemask(wav.astype(np.float64), f0, t, hp.sampling_rate)
    f0 = f0[:sum(duration)]
    
    # Compute mel-scale spectrogram and energy
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(torch.FloatTensor(wav))
    mel_spectrogram = mel_spectrogram[:, :sum(duration)]
    energy = energy[:sum(duration)]

    f0, energy = remove_outlier(f0), remove_outlier(energy)
    f0, energy = average_by_duration(f0, duration), average_by_duration(energy, duration)
    
    if mel_spectrogram.shape[1] >= hp.max_seq_len:
        # 이렇게 아예 누락하는 이유는 wav를 자르면 text도 잘린 부분만큼 처리를 해줘야하는데 그게 어려워서
        return None
    
    # Save alignment
    ali_filename = '{}-ali-{}.npy'.format(hp.dataset, basename)
    temp_dir = os.path.join(out_dir, 'alignment', subfolder)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    np.save(os.path.join(temp_dir, ali_filename), duration, allow_pickle=False)# if not os.path.isfile(os.path.join(temp_dir, ali_filename)) else None

    # Save fundamental prequency
    f0_filename = '{}-f0-{}.npy'.format(hp.dataset, basename)
    temp_dir = os.path.join(out_dir, 'f0', subfolder)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    np.save(os.path.join(temp_dir, f0_filename), f0, allow_pickle=False)# if not os.path.isfile(os.path.join(temp_dir, f0_filename)) else None

    # Save energy
    energy_filename = '{}-energy-{}.npy'.format(hp.dataset, basename)
    temp_dir = os.path.join(out_dir, 'energy', subfolder)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    np.save(os.path.join(temp_dir, energy_filename), energy, allow_pickle=False)# if not os.path.isfile(os.path.join(temp_dir, energy_filename)) else None

    # Save spectrogram
    mel_filename = '{}-mel-{}.npy'.format(hp.dataset, basename)
    temp_dir = os.path.join(out_dir, 'mel', subfolder)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    np.save(os.path.join(temp_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)# if not os.path.isfile(os.path.join(temp_dir, mel_filename)) else None
   
    if not is_valid:    # train set에만 fit해 있어야 모델의 성능이 더 좋다고 함. valid도 하면 data leakage 발생
        mel_scaler, f0_scaler, energy_scaler = scalers
        mel_scaler.partial_fit(mel_spectrogram.T)

        '''# 데이터 이상 유무 확인
        if sum(f0) == 0:
            print('f0 왜 이럼? ', basename)
        if sum(energy) == 0:
            print('energy 왜 이럼? ',basename)'''
        f0_scaler.partial_fit(f0[f0!=0].reshape(-1, 1))
        energy_scaler.partial_fit(energy[energy != 0].reshape(-1, 1))

    return '|'.join([basename, text]), mel_spectrogram.shape[1]
