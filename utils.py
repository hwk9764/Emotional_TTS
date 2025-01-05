import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from scipy.io import wavfile
from vocoder.vocgan_generator import Generator
import hparams as hp
import os
import text

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_alignment(tier):    # wav의 말의 시작과 끝을 찾아 return
    sil_phones = ['sil', 'sp', 'spn']

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trimming leading silences
        if phones == []:
            if p in sil_phones: # 맨 처음부터 쭉 보면서 silence면 sil_phones에 담음
                continue
            else:   # 묵음이 아닌 소리가 나왔을 때 여기가 시작임을 명시
                start_time = s
        if p not in sil_phones: # 시작 시간부터 끝 시간까지 묵음이 아닌 소리를 phones에 담음
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            phones.append(p)
        # e*sampling_rate/hop_length, s*sampling_rate/hop_length는 stft의 frame index를 나타냄
        # 각 phone이 몇 번째 frame부터 몇 번째 frame까지 지속되는지를 duration에 저장
        # frame 간격이 hop_length이기 때문에 전체 sample 수를 hop_length로 나누면 frame 수가 됨 (이해 안되면 그림 그려서 생각해보기)
        durations.append(int(e*hp.sampling_rate/hp.hop_length)-int(s*hp.sampling_rate/hp.hop_length))

    # Trimming tailing silences (맨 뒤 쪽에 존재하는 묵음을 제거)
    phones = phones[:end_idx]
    durations = durations[:end_idx]
    
    return phones, np.array(durations), start_time, end_time

# meta_path 파일을 열고 한 줄씩 읽으며 name(파일 이름), text(음성의 정답 라벨)를 분리하는 작업
def process_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        text = []
        name = []
        for line in f.readlines():
            n, t = line.strip('\n').split('|')
            name.append(n)
            text.append(t)
        return name, text

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def plot_data(data, titles=None, filename=None):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]

    def add_axis(fig, old_ax, offset=0):
        ax = fig.add_axes(old_ax.get_position(), anchor='W')
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        spectrogram, pitch, energy = data[i]
        axes[i][0].imshow(spectrogram, origin='lower')
        axes[i][0].set_aspect(2.5, adjustable='box')
        axes[i][0].set_ylim(0, hp.n_mel_channels)
        axes[i][0].set_title(titles[i], fontsize='medium')
        axes[i][0].tick_params(labelsize='x-small', left=False, labelleft=False) 
        axes[i][0].set_anchor('W')
        
        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(expand_by_duration(pitch), color='tomato')
        ax1.set_xlim(0, spectrogram.shape[1])
        ax1.set_ylim(0, hp.f0_max)
        ax1.set_ylabel('F0', color='tomato')
        ax1.tick_params(labelsize='x-small', colors='tomato', bottom=False, labelbottom=False)
        
        ax2 = add_axis(fig, axes[i][0], 1.2)
        ax2.plot(expand_by_duration(energy), color='darkviolet')
        ax2.set_xlim(0, spectrogram.shape[1])
        ax2.set_ylim(hp.energy_min, hp.energy_max)
        ax2.set_ylabel('Energy', color='darkviolet')
        ax2.yaxis.set_label_position('right')
        ax2.tick_params(labelsize='x-small', colors='darkviolet', bottom=False, labelbottom=False, left=False, labelleft=False, right=True, labelright=True)
    
    plt.savefig(filename, dpi=200)
    plt.close()

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item() # length 원소 중 가장 큰 값. 이걸 기준으로 padding할 것

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)

    mask = (ids >= lengths.unsqueeze(1).expand(-1, max_len))

    return mask

def get_vocgan(ckpt_path, n_mel_channels=hp.n_mel_channels, generator_ratio = [4, 4, 2, 2, 2, 2], n_residual_layers=4, mult=256, out_channels=1):

    checkpoint = torch.load(ckpt_path, map_location=torch.device(device))
    model = Generator(n_mel_channels, n_residual_layers,
                        ratios=generator_ratio, mult=mult,
                        out_band=out_channels)

    model.load_state_dict(checkpoint['model_g'])
    model.to(device).eval()

    return model

def vocgan_infer(mel, vocoder, path):
    model = vocoder

    with torch.no_grad():
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)

        audio = model.infer(mel).squeeze()
        audio = hp.max_wav_value * audio[:-(hp.hop_length*10)]
        audio = audio.clamp(min=-hp.max_wav_value, max=hp.max_wav_value-1)
        audio = audio.short().cpu().detach().numpy()

        wavfile.write(path, hp.sampling_rate, audio)


def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded

def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output

def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0)for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len-batch.size(0)), "constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

# from dathudeptrai's FastSpeech2 implementation
def standard_norm(x, mean, std, is_mel=False):

    if not is_mel:
        x = remove_outlier(x)

    zero_idxs = np.where(x == 0.0)[0]
    x = (x - mean) / std
    x[zero_idxs] = 0.0
    return x

def standard_norm_torch(x, mean, std):
    zero_idxs = torch.where(x == 0.0)[0]
    x = (x - mean) / std
    x[zero_idxs] = 0.0
    return x


def de_norm(x, mean, std):
    zero_idxs = torch.where(x == 0.0)[0]
    x = mean + std * x
    x[zero_idxs] = 0.0
    return x


def _is_outlier(x, lower, upper):
    """Check if value is an outlier."""
    return np.logical_or(x < lower, x > upper)    # outlier인지(False) 정상값인지(True) boolean list를 만듦


def remove_outlier(x):
    """Remove outlier from x.
    IQR로 이상치를 구별"""
    # 무성음/묵음 구간의 값이 0으로 표현되는데 IQR을 하게 되면 이 부분에 값이 생겨버려서 좋지 않은 영향을 끼칠 수 있음
    # non-zero 값에 대해서만 IQR을 구하고 clipping
    non_zeros = x[x!=0]
    p25 = np.percentile(non_zeros, 25)
    p75 = np.percentile(non_zeros, 75)
    lower = p25 - 1.5 * (p75 - p25) # 하단 수염(whisker)
    upper = p75 + 1.5 * (p75 - p25) # 상단 수염

    x[x!=0] = np.clip(x[x!=0], lower, upper)  # x를 lower와 upper 사이의 값으로 제한 (0값 빼고)
    return x

def average_by_duration(x, durs):
    '''
    이 코드는 각 문자에 대응되는 f0와 energy 값을 구하는 코드. TTS에서는 모델이 각 문자가 어떻게 소리되는지를 학습해야 함
    하지만 초기 f0나 energy 값들은 frame 단위로 존재하기 때문에 보다 자연스러운 소리를 학습할 수 있겠지만 각 글자가 어떻게 발음되는지 학습하기 어려울 수 있음
    그래서 이 코드에서 각 글자별로 f0, energy를 하나의 값으로 frame들의 값들을 통일시키는 것.
    '''
    mel_len = durs.sum()    # 오디오 전체 길이
    durs_cum = np.cumsum(np.pad(durs, (1, 0)))  # np.pad : durs의 왼쪽으로 1개, 오른쪽으로 0개 padding을 추가한다
    # np.cumsum : cumulative sum을 함. a라는 배열이 있고 b=np.cumsum(a)면 b_i = a_{i-1} + a_i

    # calculate charactor f0/energy
    # x_char : 문자 별 f0/energy 값
    x_char = np.zeros((durs.shape[0],), dtype=np.float32)
    for idx, start, end in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
        values = x[start:end][np.where(x[start:end] != 0.0)[0]]
        x_char[idx] = np.mean(values) if len(values) > 0 else 0.0  # np.mean([]) = nan.

    return x_char.astype(np.float32)

def expand_by_duration(x_char, durs):
    """
    오로지 시각화를 위한 메소드
    문자 단위의 값을 프레임 단위로 확장
    x_char: 문자별 값 (f0 또는 energy)
    durs: 각 문자의 지속 시간(프레임 수)
    """
    mel_len = durs.sum()
    x_frame = np.zeros((mel_len,), dtype=np.float32)
    
    current_idx = 0
    for char_idx, dur in enumerate(durs):
        x_frame[current_idx:current_idx + dur] = x_char[char_idx]
        current_idx += dur
        
    return x_frame