# Korean FastSpeech 2 - Pytorch Implementation

# Introduction

최근 딥러닝 기반 음성합성 기술이 발전하며, 자기회귀적 모델의 느린 음성 합성 속도를 개선하기 위해 비자기회귀적 음성합성 모델이 제안되었습니다. <br>
FastSpeech2는 비자기회귀적 음성합성 모델들 중 하나로, Montreal Forced Aligner(M. McAuliffe et.al., 2017)에서 phoneme(text)-utterance alignment를 추출한 duration 정보를 학습하고, 이를 바탕으로 phoneme별 duration을 예측합니다.<br>
예측된 duration을 바탕으로 phoneme-utterance alignment가 결정되고 이를 바탕으로 phoneme에 대응되는 음성이 생성됩니다. 그러므로, FastSpeech2를 학습시키기 위해서는 MFA에서 학습된 phoneme-utterance alignment 정보가 필요합니다.

이 프로젝트는 Microsoft의 [**FastSpeech 2(Y. Ren et. al., 2020)**](https://arxiv.org/abs/2006.04558)를 [**AIHub 감성 및 발화 스타일별 음성합성 데이터 (이하 emotion dataset)**](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=466)에서 동작하도록 구현한 것입니다.
본 프로젝트에서 사용된 데이터는 emotion dataset 전체가 아닌, 10명의 화자(1, 2, 4, 7, 14, 20, 53, 55, 59, 62)와 4개의 감정(분노, 슬픔, 기쁨, 무감정)에 해당하는 데이터 23,600개(약 30시간)만 골라내 수집하였습니다.

본 프로젝트에서는 아래와 같은 contribution을 제공합니다.
* emotion dataset에 대해 동작하게 만든 소스코드
* Montreal Forced Aligner로부터 추출한 emotion dataset의 text-utterance duration 정보 (TextGrid)
* emotion dataset에 대해 학습한 FastSpeech2(Text-to-melspectrogram network) pretrained model
* 화자 정보와 감정 정보를 모델링하는 임베딩 레이어를 추가

# Install Dependencies

먼저 gcc와 g++를 설치합니다.
```
# ffmpeg install
sudo apt-get install gcc g++
```

다음으로, 필요한 모듈을 pip를 이용하여 설치합니다.
```
pip install -r requirements.txt
```
# Preprocessing

**(1) emotion dataset download**
* [AIHub - emotion dataset](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=466): 약 1,076시간의 샘플로 구성된 한국어 다화자 감정발화 dataset입니다.

dataset을 다운로드 하신 후, 압축을 해제하시고 ``hparams.py``에 있는 ``data_path``에 다운받은 emotion dataset의 경로를 기록해주세요.

**(2) phoneme-utterance sequence간 alignment 정보 download**

FastSpeech2를 학습하기 위해서는 [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/)(MFA)에서 추출된 utterances와 phoneme sequence간의 alignment가 필요합니다. MFA 학습 방법은 [다음 링크](https://github.com/JH-lee95/Fastspeech2-Korean?tab=readme-ov-file)에서 참고하시면 됩니다. 생성한 ```TextGrid.zip```파일을 ``프로젝트 폴더 (Korean-FastSpeech2-Pytorch)``에 두세요. 

**(3) 데이터 전처리**
```
python preprocess.py
```
data 전처리를 위해 위의 커맨드를 입력해 주세요. 전처리 된 데이터는 프로젝트 폴더의 ``preprocessed/`` 폴더에 생성됩니다.

    
# Train
모델 학습 전에, kss dataset에 대해 사전학습된 VocGAN(neural vocoder) [다운로드](https://drive.google.com/file/d/1GxaLlTrEhq0aXFvd_X1f4b-ev7-FH8RB/view?usp=sharing)과 HiFi-GAN [다운로드](https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y)하여 ``vocoder/pretrained_models/`` 경로에 위치시킵니다.

다음으로, 아래의 커맨드를 입력하여 모델 학습을 수행합니다.
```
python train.py
```
학습된 모델은 ``ckpt/``에 저장되고 tensorboard log는 ``log/``에 저장됩니다. 학습시 evaluate 과정에서 생성된 음성은 ``eval/`` 폴더에 저장됩니다.

# Synthesis
학습된 파라미터를 기반으로 음성을 생성하는 명령어는 다음과 같습니다. 
```
python synthesis.py --step 500000
```
합성된 음성은  ```results/``` directory에서 확인하실 수 있습니다.

# Tensorboard
```
tensorboard --logdir log/
```
tensorboard log들은 ```log/hp.dataset/``` directory에 저장됩니다. 그러므로 위의 커멘드를 이용하여 tensorboard를 실행해 학습 상황을 모니터링 하실 수 있습니다.

- 학습 과정 시각화
![Image](https://github.com/user-attachments/assets/8ff7576d-51d1-4669-bffb-9caffcbb8307)
![Image](https://github.com/user-attachments/assets/a2f3a052-2d7d-471e-acd8-e0f30e741840)

- 합성시 생성된 melspectrogram과 예측된 f0, energy values
![Image](https://github.com/user-attachments/assets/ae31768b-19d0-4465-be6d-9044063212f0)


# References
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), Y. Ren, *et al*.
- [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263), Y. Ren, *et al*.
- [ming024's FastSpeech2 impelmentation](https://github.com/ming024/FastSpeech2)
- [rishikksh20's VocGAN implementation](https://github.com/rishikksh20/VocGAN)
- [HGU-DLLAB](https://github.com/HGU-DLLAB/Korean-FastSpeech2-Pytorch?tab=readme-ov-file)
- [JH-lee95](https://github.com/JH-lee95/Fastspeech2-Korean?tab=readme-ov-file)
