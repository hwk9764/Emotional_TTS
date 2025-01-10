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

### set GPU number ###
train_visible_devices = "0"
synth_visible_devices = "0"

# Text
text_cleaners = ['korean_cleaners']

# Audio and mel
### kss ###
sampling_rate = 22050
filter_length = 512    # n_fft
hop_length = 128
win_length = 512

### kss ###
max_wav_value = 32768.0
n_mel_channels = 80 # == gst의 hp.n_mels
mel_fmin = 0
mel_fmax = 8000

f0_min = 50
f0_max = 300
energy_min = 0.0
energy_max = 150

# GST
ref_enc_filters = [32, 32, 64, 64, 128, 128]    # ref_enc의 6개 conv layer의 각 output channel의 수
n_mels = 80
E = 256 # style token embedding의 차원. text encoder(TTS의 encoder)의 input 차원 수에 맞춤.
token_num = 10  # 감정 정보를 구하는 데 사용되는 style token의 개수. 논문에선 10개가 적당하다는 결론을 얻었다고 얘기함. style token을 weighted sum해서 style embedding을 구함.
num_heads = 8


# FastSpeech 2
encoder_layer = 5
encoder_head = 4
encoder_hidden = 256    # encoder input의 embedding 사이즈
decoder_layer = 5
decoder_head = 4
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
# 근데 실제로는 최대 길이가 3500이 넘지 않음. 어떻게 구하는지 모르겠다.
max_seq_len = 3500

# Optimizer
batch_size = 8
accumulate_steps = 4   # 16 * 3 -> fastspeech2 논문과 동일
epochs = 1000
step_per_epoch = 23597/(batch_size*accumulate_steps) # 23597은 데이터셋 개수
n_warm_up_step = 3000 #int(epochs*step_per_epoch*0.02) # 데이터 수에 따라 조절
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
f_loss_weight = 4.0
e_loss_weight = 4.0

# Vocoder
vocoder = 'vocgan'
vocoder_pretrained_model_name = "vocgan_kss_pretrained_model_epoch_4500(1).pt"
vocoder_pretrained_model_path = "vocoder/pretrained_models/vocgan_kss_pretrained_model_epoch_4500(1).pt"

# Log-scaled duration
log_offset = 1.

# Save, log and synthesis
save_step = int(step_per_epoch)*10
eval_step = int(step_per_epoch)*5
log_step = int(step_per_epoch)*10
clear_Time = 20
restore_step = 0
synthesize_step = 193500