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
from vocoder import vocgan_generator

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

    if vocoder is not None:
        if hp.vocoder.lower() == "vocgan":
            utils.vocgan_infer(mel_postnet_torch, vocoder, path=os.path.join(hp.test_path, '{}_{}_{}.wav'.format(prefix, sentence, hp.vocoder)))
    
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
        vocoder = utils.get_vocgan(ckpt_path=hp.vocoder_pretrained_model_path)
    else:
        vocoder = None   
 
    #kss
    eval_sentence=['그는 괜찮은 척하려고 애쓰는 것 같았다','그녀의 사랑을 얻기 위해 애썼지만 헛수고였다','용돈을 아껴써라','그는 아내를 많이 아낀다','요즘 공부가 안돼요','한 여자가 내 옆에 앉았다']
    train_sentence=['가까운 시일 내에 한번, 댁으로 찾아가겠습니다','우리의 승리는 기적에 가까웠다','아이들의 얼굴에는 행복한 미소가 가득했다','헬륨은 공기보다 가볍다','이것은 간단한 문제가 아니다']
    test_sentence=['안녕하세요, 한동대학교 딥러닝 연구실입니다.', '이 프로젝트가 여러분에게 도움이 되었으면 좋겠습니다.', '시간이 촉박해요','이런, 큰일 났어','좀 더 먹지 그래?','제가 뭘 잘못했죠?','더 이상 묻지마']
    
    g2p=G2p()
    print('which sentence do you want?')
    print('1.eval_sentence 2.train_sentence 3.test_sentence 4.create new sentence')

    mode=input()
    print('you went for mode {}'.format(mode))
    if mode=='4':
        print('input sentence')
        sentence = input()
    elif mode=='1':
        sentence = eval_sentence
    elif mode=='2':
        sentence = train_sentence
    elif mode=='3':
        sentence = test_sentence
    
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
    if mode != '4':
        for s in sentence:
            text = kor_preprocess(s)
            synthesize(model, vocoder, text, s, s_id, e_id, dur_pitch_energy_aug, prefix='{}-step_{}-duration_{}-pitch_{}-energy_{}-speaker_{}-emotion_{}'.format(date_format, args.step, dur_pitch_energy_aug[0], dur_pitch_energy_aug[1], dur_pitch_energy_aug[2], speaker, emotion))
    else:
        text = kor_preprocess(sentence)
        synthesize(model, vocoder, text, sentence, s_id, e_id, dur_pitch_energy_aug, prefix='{}-step_{}-duration_{}-pitch_{}-energy_{}-speaker_{}-emotion_{}'.format(date_format, args.step, dur_pitch_energy_aug[0], dur_pitch_energy_aug[1], dur_pitch_energy_aug[2], speaker, emotion))
