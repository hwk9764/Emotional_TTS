import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import Encoder, Decoder
from transformer.Layers import PostNet
from modules import VarianceAdaptor
from utils import get_mask_from_lengths
import hparams as hp
from GST import GST

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, use_postnet=True):
        super(FastSpeech2, self).__init__()

        self.encoder = Encoder()
        self.variance_adaptor = VarianceAdaptor()

        self.decoder = Decoder()
        self.mel_linear = nn.Linear(hp.decoder_hidden, hp.n_mel_channels)
        
        self.use_postnet = use_postnet
        if self.use_postnet:
            self.postnet = PostNet()

        self.gst = GST()

    def forward(self, src_seq, src_len, ref_mels, mel_len=None, d_target=None, p_target=None, e_target=None, max_src_len=None, max_mel_len=None, dur_pitch_energy_aug=None, f0_stat=None, energy_stat=None):
        src_mask = get_mask_from_lengths(src_len, max_src_len)  # padding을 가리는 masking
        mel_mask = get_mask_from_lengths(mel_len, max_mel_len) if mel_len is not None else None # 미래 mel token을 가리는 masking
        
        encoder_output = self.encoder(src_seq, src_mask)
        # encoder에 masking을 사용하는 이유
        # input들이 모두 길이가 다르기 때문에 모델에 넣기 전에 padding으로 길이를 맞춰준다.
        # 이 padding에 attention하면 바람직한 결과가 나오지 않을 것이기 때문에 이 padding에 masking을 하는 것이 source masking이다.
        # 좀 더 정확히는 softmax 때 문제가 발생하므로 pad token과 attention은 할 수 있지만 softmax에서 영향을 발휘하지 않게 하기 위해 softmax 직전에 masking을 함.
        style_embed = self.gst(ref_mels)  # [N, 256]
        #style_embed = style_embed.expand_as(encoder_output)    # expand나 expand_as나 똑같은데 코드의 유연성과 재사용성을 위해 expand가 더 적합하다고 함.
        style_embed = style_embed.expand(-1, encoder_output.size(1), -1)
        encoder_output = encoder_output + style_embed
        
        # train
        if d_target is not None:
            variance_adaptor_output, d_prediction, p_prediction, e_prediction, _, _ = self.variance_adaptor(
                encoder_output, src_mask, mel_mask, d_target, p_target, e_target, max_mel_len)
        # inference
        else:
            variance_adaptor_output, d_prediction, p_prediction, e_prediction, mel_len, mel_mask = self.variance_adaptor(
                    encoder_output, src_mask, mel_mask, d_target, p_target, e_target, max_mel_len, dur_pitch_energy_aug, f0_stat, energy_stat)
        # Fastspeech2의 decoder는 원래 target input이 들어가지 않고, encoder output과 variance 정보가 합쳐진 input만 들어간다.
        # decoder는 mel-spectrogram을 예측하는 것이기 때문에 mel_mask로 미래 mel값을 가리면서 self-attention을 수행한다. (원래 transformer에서 decoder input에 하는 것과 동일)
        decoder_output = self.decoder(variance_adaptor_output, mel_mask)
        mel_output = self.mel_linear(decoder_output)    # decoder output을 mel-spectrogram으로 convert
        
        # PostNet: Five 1-d convolution with 512 channels and kernel size 5
        # Decoder가 생성한 mel-spectrogram을 더 세밀하게 보정해 음질을 향상시킴.
        # mel-spectrogram의 잔여 오차를 줄여주는 역할을 한다.
        if self.use_postnet:
            mel_output_postnet = self.postnet(mel_output) + mel_output
        else:
            mel_output_postnet = mel_output

        return mel_output, mel_output_postnet, d_prediction, p_prediction, e_prediction, src_mask, mel_mask, mel_len


if __name__ == "__main__":
    # Test
    model = FastSpeech2(use_postnet=False)
    print(model)
    print(sum(param.numel() for param in model.parameters()))
