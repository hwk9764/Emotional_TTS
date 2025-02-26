import os

# Checkpoints and paths
dataset = "emotion"
#data_path = "/content/drive/MyDrive/Emotional_TTS/emotion/emotion_dataset"
data_path = "D:/Dataset/emotion/emotion_dataset"  #"D:/Dataset/korean-single-speaker-speech-dataset/"
train_meta_name = "train.txt"	# "transcript.v.1.4.txt" or "transcript.v.1.3.txt"
val_meta_name = 'valid.txt'
textgrid_name = "TextGrid.zip"
preprocessed_path = os.path.join("./preprocessed/", dataset)
checkpoint_path = os.path.join("./ckpt/", dataset)
eval_path = os.path.join("./eval/", dataset)
log_path = os.path.join("./log/", dataset)
log_plus_path = os.path.join("./log_plus/")
test_path = "./results"
textgrid_path = "D:/Dataset/emotion/emotion_dataset/TextGrid"
num_best_model = 3

# Dataset
with open(os.path.join(preprocessed_path, train_meta_name), 'r', encoding='utf-8') as f:
    num_dataset = len(f.readlines())
### set GPU number ###
train_visible_devices = "0"
synth_visible_devices = "0"

# Text
text_cleaners = ['korean_cleaners']

# Audio and mel
sampling_rate = 22050
filter_length = 512    # n_fft
hop_length = 128
win_length = 512

### kss ###
max_wav_value = 32768.0
n_mel_channels = 80 # == gst의 hp.n_mels
mel_fmin = 0
mel_fmax = 8000

f0_min = 0.0
f0_max = 400
energy_min = 0.0
energy_max = 150

# GST
ref_enc_filters = [32, 32, 64, 64, 128, 128]    # ref_enc의 6개 conv layer의 각 output channel의 수
n_mels = 80
E = 256 # style token embedding의 차원. text encoder(TTS의 encoder)의 input 차원 수에 맞춤.
token_num = 10  # 감정 정보를 구하는 데 사용되는 style token의 개수. 논문에선 10개가 적당하다는 결론을 얻었다고 얘기함. style token을 weighted sum해서 style embedding을 구함.
num_heads = 4

# speaker, emotion embedding
n_speaker = 10
n_emotion = 4
speaker_id = {1:0, 2:1, 4:2, 7:3, 14:4, 20:5, 53:6, 55:7, 59:8, 62:9}
emotion_id = {"분노":0, "기쁨":1, "무감정":2, "슬픔":3}

# FastSpeech 2
encoder_layer = 4
encoder_head = 2
encoder_hidden = 256    # encoder input의 embedding 사이즈
decoder_layer = 6
decoder_head = 2
decoder_hidden = 256    # decoder input의 embedding 사이즈
fft_conv1d_filter_size = 1024
fft_conv1d_kernel_size = (9, 1)
encoder_dropout = 0.2
decoder_dropout = 0.2

variance_predictor_filter_size = 256
variance_predictor_kernel_size = 3
variance_predictor_dropout = 0.5

# 30초 이상은 자른다고 할 때 mel_spec의 최대 seq_len은 sampling_rate * sec(오디오 길이) / hop_length (frame 간격)
# 22050 * 30 / 128 ≒ 5168
# 오디오를 자르게 되면 그에 맞춰 label인 text도 잘라야 하는데 이 과정이 귀찮으므로 자르지 않도록 크게 잡는 것이 편함
max_seq_len = 2000

# Optimizer
batch_size = 48
accumulate_steps = 1
epochs = 500
step_per_epoch = num_dataset/(batch_size*accumulate_steps)
n_warm_up_step = 4000 #int(epochs*step_per_epoch*0.02) # 데이터 수에 따라 조절
anneal_step = 300000
anneal_rate = 0.3
grad_clip_thresh = 1.0
learning_rate = 7e-5
betas = (0.9, 0.98)
eps = 1e-9
weight_decay = 0.1
early_stop = 30

# Loss weight
mel_loss_weight = 1.0
mel_postnet_loss_weight = 1.0
d_loss_weight = 1.0
f_loss_weight = 1.0
e_loss_weight = 1.0

# Vocoder
vocoder = "hifigan"#'vocgan'
vocgan_pretrained_model_path = "vocoder/pretrained_models/vocgan_kss_pretrained_model_epoch_4500(1).pt"
hifigan_pretrained_model_path = "vocoder/pretrained_models/g_02500000"

# Log-scaled duration
log_offset = 1.

# Save, log and synthesis
save_step = int(step_per_epoch)*50
eval_step = int(step_per_epoch)*50
log_step = int(step_per_epoch)*50
clear_Time = 20
restore_step = 0
synthesize_step = 193500