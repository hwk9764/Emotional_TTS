import torch
import torch.nn as nn
import numpy as np
import hparams as hp
import os
import random
from datetime import datetime

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=hp.synth_visible_devices

import argparse
import re
from string import punctuation

from fastspeech2 import FastSpeech2
from vocoder import vocgan_generator, hifigan_generator

from text import text_to_sequence, sequence_to_text
import utils
import audio as Audio

from scipy.io.wavfile import read
import librosa
from g2pk import G2p
from jamo import h2j

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

def kor_preprocess(text):
    text = re.sub(r'[,!“”‘’〈〉《》「」『』…´`]', '', text)
    text = re.sub(r'[·/-]', ' ', text)
    g2p=G2p()
    phone = g2p(text)
    print('after g2p: ',phone)
    phone = h2j(phone)
    print('after h2j: ',phone)
    phone = list(filter(lambda p: p != ' ', phone))
    phone = '{' + '}{'.join(phone) + '}'
    print('phone: ',phone)
    phone = re.sub(r'\{[^\w\s]?\}', '{sil}', phone)
    print('after re.sub: ',phone)
    phone = phone.replace('}{', ' ')

    print('|' + phone + '|')
    sequence = np.array(text_to_sequence(phone,hp.text_cleaners))
    sequence = np.stack([sequence])
    return torch.from_numpy(sequence).long().to(device)

def get_FastSpeech2(num):
    checkpoint_path = os.path.join(hp.checkpoint_path, "checkpoint_{}.pth.tar".format(num))
    model = nn.DataParallel(FastSpeech2())
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.requires_grad = False
    model.eval()
    return model

def synthesize(model, vocoder, text, sentence, ref_path, dur_pitch_energy_aug, prefix=''):
    '''arguments
    model : FastSpeech2
    vocoder : VocGAN
    text : 원문
    sentence : 원문 g2p
    dur_pitch_energe_aug : duration 가중치
    prefix : 합성 음성 파일 이름 형식
    '''
    sentence = sentence[:10] # long filename will result in OS Error

    mean_mel, std_mel = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "mel_stat.npy")), dtype=torch.float).to(device)
    mean_f0, std_f0 = f0_stat = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "f0_stat.npy")), dtype=torch.float).to(device)
    mean_energy, std_energy = energy_stat = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "energy_stat.npy")), dtype=torch.float).to(device)

    mean_mel, std_mel = mean_mel.reshape(1, -1), std_mel.reshape(1, -1)
    mean_f0, std_f0 = mean_f0.reshape(1, -1), std_f0.reshape(1, -1)
    mean_energy, std_energy = mean_energy.reshape(1, -1), std_energy.reshape(1, -1)

    src_len = torch.from_numpy(np.array([text.shape[1]])).to(device)
    
    wav, _ = librosa.load(ref_path)

    mel_spectrogram, _ = Audio.tools.get_mel_from_wav(torch.FloatTensor(wav))
    mel_spectrogram = torch.from_numpy(mel_spectrogram.numpy().astype(np.float32).T).unsqueeze(0)
    ref_mels = mel_spectrogram[:, 1:, :]
    
    mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = model(text, src_len, ref_mels, dur_pitch_energy_aug=dur_pitch_energy_aug, f0_stat=f0_stat, energy_stat=energy_stat)    

    mel_torch = mel.transpose(1, 2).detach()
    mel_postnet_torch = mel_postnet.transpose(1, 2).detach()
    f0_output = f0_output[0]
    energy_output = energy_output[0]

    mel_torch = utils.de_norm(mel_torch.transpose(1, 2), mean_mel, std_mel)
    mel_postnet_torch = utils.de_norm(mel_postnet_torch.transpose(1, 2), mean_mel, std_mel).transpose(1, 2)
    D_output = (np.exp(log_duration_output[0].detach().cpu().numpy())-1).astype(np.int64)
    f0_output = utils.de_norm(f0_output, mean_f0, std_f0).squeeze().detach().cpu().numpy()
    energy_output = utils.de_norm(energy_output, mean_energy, std_energy).squeeze().detach().cpu().numpy()

    if not os.path.exists(hp.test_path):
        os.makedirs(hp.test_path)

    Audio.tools.inv_mel_spec(mel_postnet_torch[0], os.path.join(hp.test_path, '{}_{}_griffin_lim.wav'.format(prefix, sentence)))

    if hp.vocoder.lower() == "vocgan":
        utils.vocoder_infer(mel_postnet_torch, vocoder, path=os.path.join(hp.test_path, '{}_{}_{}.wav'.format(prefix, sentence, hp.vocoder)))
    
    utils.plot_data([(mel_postnet_torch[0].detach().cpu().numpy(), f0_output, energy_output)], titles=['Synthesized Spectrogram'], filename=os.path.join(hp.test_path, '{}_{}.png'.format(prefix, sentence)))


if __name__ == "__main__":
    year = str(datetime.now().year)
    month = datetime.now().month if datetime.now().month>9 else '0'+str(datetime.now().month)
    day = datetime.now().day if datetime.now().day>9 else '0'+str(datetime.now().day)
    hour = datetime.now().hour if datetime.now().hour>9 else '0'+str(datetime.now().hour)
    minute = datetime.now().minute if datetime.now().minute>9 else '0'+str(datetime.now().minute)
    date_format = f'{year}{month}{day}-{hour}{minute}'
    
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=30000)
    args = parser.parse_args()

    # duration, pitch, energy를 조절. 입력한 파라미터와 곱해서 크기를 조절 ex) duration 값이 1보다 작으면 더 빨리 말하게 할 수 있음
    dur_pitch_energy_aug = [1.0, 1.0, 1.0]    	# [duration, pitch, energy]

    model = get_FastSpeech2(args.step).to(device)
    if hp.vocoder == 'vocgan':
        vocoder = utils.get_vocoder(ckpt_path=hp.vocgan_pretrained_model_path)
    elif hp.vocoder == "hifigan":
        vocoder = utils.get_vocoder(ckpt_path=hp.hifigan_pretrained_model_path)
 
    #kss
    short_sentence=['안녕하세요, 한동대학교 딥러닝 연구실입니다.','이런, 큰일 났어.','제가 뭘 잘못했죠?','더 이상 묻지마.','그렇게 말해주다니 너무 고마워.', '나는 그러려던 게 아니었는데...']
    long_sentence=["매일 똑같은 나의 하루 속에 가끔은 내게 선물이 돼줄 그런 사람이 있었으면 해. 달이 예쁘다고 무슨 일은 없었냐고 물어봐줄 그런 사람 말야. 그리고 그런 사람이 너였으면 해.",
                   "나 스무살 적에 하루를 견디고 불안한 잠자리에 누울 때면 '내일 뭐하지?' 걱정을 했어. 눈을 감아도 통 잠은 안오고 가슴은 답답할 때 난 왜 안되는지를 되뇌었어. 그러던 어느 날 내 맘에 찾아온 작지만 놀라운 깨달음이 내일 뭘 할지 꿈꾸게 했지. 사실 한 번도 미친 듯이 그렇게 달려든 적이 없었다는 것을 생각해 봤고 내 자신을 일으켜 세웠어.",
                   "한때는 서로의 세상이었던 우리가 이제는 아무 말 없이 스쳐 지나가는 타인이 되어버렸다는 사실이 믿기지 않아, 밤이 깊어질수록 가슴 한편에 남아있는 미련과 후회가 뒤엉켜 나를 잠 못 이루게 하고, 그리움은 끝내 눈물이 되어 조용히 베개를 적셔가지만, 아무리 애써도 되돌릴 수 없는 시간 앞에서 무력함만이 나를 짓누르고 있을 뿐이다.",
                   "믿었던 만큼 배신감이 날카롭게 가슴을 찢어놓고, 너의 거짓말과 변명들이 내 귀를 울릴 때마다 끓어오르는 분노가 나를 삼키며, 도대체 무엇이 진실이었는지조차 알 수 없게 만들어버린 네가 참으로 증오스럽다."]
    
    g2p=G2p()
    print('1.short_sentence 2.long_sentence 3.create new sentence')

    mode=input()
    print('you went for mode {}'.format(mode))
    if mode=='3':
        print('input sentence')
        sentence = input()
    elif mode=='1':
        sentence = random.sample(short_sentence)
    elif mode=='2':
        sentence = random.sample(long_sentence)
    
    print('sentence that will be synthesized: ', sentence)
    print('what speaker do you want? choose between (1, 2, 4, 7, 14, 20, 53, 55, 59, 62)')
    while True:
        speaker=int(input())
        if hp.speaker_id.get(speaker) != None:
            print('you choosed ', speaker)
            s_id = hp.speaker_id[speaker]
            break
        print('input proper value')
    print('what emotion do you want? choose between (분노, 기쁨, 무감정, 슬픔)')
    while True:
        emotion=input()
        if hp.emotion_id.get(emotion) != None:
            print('you choosed ', emotion)
            e_id = hp.emotion_id[emotion]
            break
        print('input proper value')

    text = kor_preprocess(sentence)
    synthesize(model, vocoder, text, sentence, s_id, e_id, dur_pitch_energy_aug, prefix='{}-step_{}-duration_{}-pitch_{}-energy_{}-speaker_{}-emotion_{}'.format(date_format, args.step, dur_pitch_energy_aug[0], dur_pitch_energy_aug[1], dur_pitch_energy_aug[2], speaker, emotion))