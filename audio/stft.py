import torch
import torch.nn.functional as F
import numpy as np

from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn

from audio.audio_processing import dynamic_range_compression
from audio.audio_processing import dynamic_range_decompression
from audio.audio_processing import window_sumsquare

# STFT를 그냥 하지 않고 이렇게 학습 가능하게 구현하는 이유는 현재 모델과 task에 맞는 최적의 window size, n_fft size를 학습해서 찾기 위함
# Batch Norm, Layer Norm과 똑같은 이유라고 보면 됨
# https://dsp.stackexchange.com/questions/86937/equivalence-between-windowed-fourier-transform-and-stft-as-convolutions-filter

class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, filter_length, hop_length, win_length,
                 window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))  # self.filter_length 크기의 정방단위행렬 생성하여 fft

        # cutoff 설명
        # DFT(FFT)는 discrete sample들을 FT하기 때문에 FT 후에도 sample 수는 유지됨
        # 그런데 변환된 신호가 -T/2 ~ T/2 구간의 신호가 무한히 반복되는 구조라서 T/2 ~ T까지는 짜름
        # 그래서 filter_length/2+1하는 것. (+1은 0Hz인 DC 성분과 Nyquist 주파수(최대 주파수)를 포함하기 위함)
        cutoff = int((self.filter_length / 2 + 1))
        
        # fourier_basis 설명
        # mel filter bank처럼 여러 개의 filter가 모여 있는 집합체 (아직은 window가 아님. 굳이 따지자면 rect함수)
        # STFT를 할 때 sliding window를 하는 게 아니라 filter들과 원 신호를 곱해서 합치면 sliding window로 STFT한 것과 같은 결과가 나옴
        # 그러니까 병렬 처리한다고 생각하면 됨 
        # FFT 결과는 복소수로 이루어져있는데 Convolution을 통과시키기 위해서는 실수부와 허수부를 따로 계산해야 하므로 나눠줌
        # 분리하여 vstack으로 쌓은 후 Convolution을 하면 실수부끼리, 허수부끼리 연산이 된다. (∵ 1d conv라서)
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])    # Conv에 넣을 수 있는 형태로 변환. 그냥 unsqueeze(1)과 똑같은 동작
        
        # inverse_basis 설명
        # inverse FFT를 위한 filter의 집합체
        # fourier_basis를 inverse FFT하여 구하하
        # STFT 계산 시 스케일링 효과가 발생하기 때문에 inverse STFT시 이를 보정함
        # scale은 filter_length / hop length로 설정된다.
        # np.linalg.pinv는 역행렬을 구하는 건데 이게 바로 inverse STFT하는 것것
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            # window랑 곱해서 각 필터를 window로 변환. 이렇게 한 후 conv를 하게 되면 STFT하는 것과 같은 모양이 된다.
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def transform(self, input_data):
        # STFT 하는 메소드
        # input_data : wav
        num_batches = input_data.size(0)    # 1
        num_samples = input_data.size(1)    # wav의 길이

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data.to(self.device),
            self.forward_basis.to(self.device),
            stride=self.hop_length,
            padding=0).cpu()

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.atan2(imag_part.data, real_part.data)

        return magnitude, phase

    def inverse(self, magnitude, phase):
        # inverse STFT하는 메소드
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.from_numpy(window_sum)
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :,
                              approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:,
                                              :, :-int(self.filter_length/2):]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length, hop_length, win_length,
                 n_mel_channels, sampling_rate, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax) # mel filter bank
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        but in preprocessing, from a wav => 1차원 waveform in, 2차원 mel-spectrogram out
        PARAMS
        ------
        y: (학습 시) torch.FloatTensor with shape (B, T) in range [-1, 1]
            (preprocess 시) torch.FloatTensor with shpae (T, )

        param B : batch size(hparams에서 batch size 보기), T : # of timestep 또는 sample의 개수(즉, waveform의 duration)

        RETURNS
        -------
        mel_output: (학습 시) torch.FloatTensor of shape (B, n_mel_channels, T)
                    (preprocess 시) torch.FloatTensor of shape (n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        energy = torch.norm(magnitudes, dim=1)
         
        return mel_output, energy
