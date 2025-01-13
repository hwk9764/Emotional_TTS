import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import hparams as hp
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=hp.train_visible_devices

import numpy as np
import argparse
import time
from fastspeech2 import FastSpeech2
from loss import FastSpeech2Loss
from dataset import Dataset
from optimizer import ScheduledOptim
from evaluate import evaluate
import utils
import audio as Audio

def main(args):
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("device :", device)
    
    # Get dataset
    dataset = Dataset("train.txt") # train.txt : kss의 transcription
    # dataset의 각 내용물은 audio 데이터의 id, text, mel, alignment, f0, energy 값을 가지고 있는 dictionary
    loader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True, 
        collate_fn=dataset.collate_fn, drop_last=True, num_workers=0, pin_memory=True)
    
    # Define model
    model = nn.DataParallel(FastSpeech2()).to(device)
    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of FastSpeech2 Parameters:', num_param)

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), betas=hp.betas, eps=hp.eps, weight_decay = hp.weight_decay)
    scheduled_optim = ScheduledOptim(optimizer, hp.decoder_hidden, hp.n_warm_up_step, args.restore_step, int(hp.epochs*hp.step_per_epoch))
    Loss = FastSpeech2Loss().to(device) 
    print("Optimizer and Loss Function Defined.")

    # Load checkpoint if exists
    checkpoint_path = os.path.join(hp.checkpoint_path)
    try:
        checkpoint = torch.load(os.path.join(
            checkpoint_path, 'checkpoint_{}.pth.tar'.format(args.restore_step)))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step {}---\n".format(args.restore_step))
    except:
        print("\n---Start New Training---\n")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

    # read params
    mean_mel, std_mel = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "mel_stat.npy")), dtype=torch.float).to(device)
    mean_f0, std_f0 = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "f0_stat.npy")), dtype=torch.float).to(device)
    mean_energy, std_energy = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "energy_stat.npy")), dtype=torch.float).to(device)

    mean_mel, std_mel = mean_mel.reshape(1, -1), std_mel.reshape(1, -1)
    mean_f0, std_f0 = mean_f0.reshape(1, -1), std_f0.reshape(1, -1)
    mean_energy, std_energy = mean_energy.reshape(1, -1), std_energy.reshape(1, -1)


    # Load vocoder
    if hp.vocoder == 'vocgan':
        vocoder = utils.get_vocgan(ckpt_path = hp.vocoder_pretrained_model_path)
        vocoder.to(device)
    else:
        vocoder = None

    # Init logger
    log_path = hp.log_plus_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        os.makedirs(os.path.join(log_path, 'train'))
        os.makedirs(os.path.join(log_path, 'validation'))
    train_logger = SummaryWriter(os.path.join(log_path, 'train'))
    val_logger = SummaryWriter(os.path.join(log_path, 'validation'))

    # Define Some Information
    Time = np.array([])
    Start = time.perf_counter()
    
    # Training
    model = model.train()
    prev_l = 100    # 이전 validation loss 저장
    early_stop = 0  # early stop counter
    eval_losses = []    # validation loss 저장
    total_step = int(hp.epochs * hp.step_per_epoch)
    for epoch in range(hp.epochs):
        # Get Training Loader
        print(f'epoch {epoch+1} start')

        for idx, batch in enumerate(loader):
            start_time = time.perf_counter()
            current_step = idx + epoch*len(loader) + 1 + args.restore_step
            
            # Get Data
            text = torch.from_numpy(batch["text"]).long().to(device)
            mel_target = torch.from_numpy(batch["mel_target"]).float().to(device)   # mel spectrogram.shape : (batch_size, num_frames, mel_bin)
            D = torch.from_numpy(batch["D"]).long().to(device)  # => duration
            log_D = torch.from_numpy(batch["log_D"]).float().to(device)
            f0 = torch.from_numpy(batch["f0"]).float().to(device)
            energy = torch.from_numpy(batch["energy"]).float().to(device)
            src_len = torch.from_numpy(batch["src_len"]).long().to(device)
            mel_len = torch.from_numpy(batch["mel_len"]).long().to(device)
            max_src_len = np.max(batch["src_len"]).astype(np.int32)
            max_mel_len = np.max(batch["mel_len"]).astype(np.int32)
            
            # Forward
            mel_output, mel_postnet_output, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _ = model(
                text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len)
            
            # Cal Loss
            mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss = Loss(
                    log_duration_output, log_D, f0_output, f0, energy_output, energy, mel_output, mel_postnet_output, mel_target, ~src_mask, ~mel_mask)
            total_loss = mel_loss + mel_postnet_loss + d_loss + f_loss + e_loss
                
            # Logger
            t_l = total_loss.item()
            m_l = mel_loss.item()
            m_p_l = mel_postnet_loss.item()
            d_l = d_loss.item()
            f_l = f_loss.item()
            e_l = e_loss.item()
            with open(os.path.join(log_path, "total_loss.txt"), "a") as f_total_loss:
                f_total_loss.write(str(t_l)+"\n")
            with open(os.path.join(log_path, "mel_loss.txt"), "a") as f_mel_loss:
                f_mel_loss.write(str(m_l)+"\n")
            with open(os.path.join(log_path, "mel_postnet_loss.txt"), "a") as f_mel_postnet_loss:
                f_mel_postnet_loss.write(str(m_p_l)+"\n")
            with open(os.path.join(log_path, "duration_loss.txt"), "a") as f_d_loss:
                f_d_loss.write(str(d_l)+"\n")
            with open(os.path.join(log_path, "f0_loss.txt"), "a") as f_f_loss:
                f_f_loss.write(str(f_l)+"\n")
            with open(os.path.join(log_path, "energy_loss.txt"), "a") as f_e_loss:
                f_e_loss.write(str(e_l)+"\n")
                
            # Backward
            # accumulate gradients
            total_loss = total_loss / hp.accumulate_steps
            total_loss.backward()
            if current_step % hp.accumulate_steps != 0 and (epoch!=hp.epochs-1 or idx!=len(loader)-1):    # 후자 조건은 마지막에 accumulate_step 보다 적은 배치가 남았어도 업데이트하기 위함 
                continue
            
            # accumulate step 반영한 진짜 current step 구하기
            if epoch==hp.epochs-1 and idx==len(loader)-1:
                if current_step%hp.accumulate_steps == 0: accumulated_current_step = total_step
                else: accumulated_current_step = total_step+1 # batch size만큼 안되는 남은 데이터 때문에 step 하나가 더 생김
            else: accumulated_current_step = current_step // hp.accumulate_steps
            
            # step 확인
            with open(os.path.join(log_path, "steps.txt"), "a") as step:
                step.write(str(current_step)+'/'+str(accumulated_current_step)+"\n")
                           
            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)
            # Update weights
            scheduled_optim.step_and_update_lr()
            scheduled_optim.zero_grad()
            
            # Print
            if accumulated_current_step % hp.log_step == 0:
                Now = time.perf_counter()

                str1 = "Epoch [{}/{}], Step [{}/{}]:".format(
                    epoch+1, hp.epochs, accumulated_current_step, total_step)
                str2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f}, F0 Loss: {:.4f}, Energy Loss: {:.4f};".format(
                    t_l, m_l, m_p_l, d_l, f_l, e_l)
                str3 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                    (Now-Start), (total_step-accumulated_current_step)*np.mean(Time))

                print("\n" + str1)
                print(str2)
                print(str3)
                
                with open(os.path.join(log_path, "log.txt"), "a") as f_log:
                    f_log.write(str1 + "\n")
                    f_log.write(str2 + "\n")
                    f_log.write(str3 + "\n")
                    f_log.write("\n")

            train_logger.add_scalar('Loss/total_loss', t_l, accumulated_current_step)
            train_logger.add_scalar('Loss/mel_loss', m_l, accumulated_current_step)
            train_logger.add_scalar('Loss/mel_postnet_loss', m_p_l, accumulated_current_step)
            train_logger.add_scalar('Loss/duration_loss', d_l, accumulated_current_step)
            train_logger.add_scalar('Loss/F0_loss', f_l, accumulated_current_step)
            train_logger.add_scalar('Loss/energy_loss', e_l, accumulated_current_step)
            train_logger.add_scalar('learning_rate', scheduled_optim._optimizer.param_groups[0]['lr'], accumulated_current_step)

            if accumulated_current_step % hp.eval_step == 0:
                model.eval()
                with torch.no_grad():
                    d_l, f_l, e_l, m_l, m_p_l = evaluate(model, accumulated_current_step, vocoder)
                    t_l = m_l + m_p_l + d_l + f_l + e_l

                    val_logger.add_scalar('Loss/total_loss', t_l, accumulated_current_step)
                    val_logger.add_scalar('Loss/mel_loss', m_l, accumulated_current_step)
                    val_logger.add_scalar('Loss/mel_postnet_loss', m_p_l, accumulated_current_step)
                    val_logger.add_scalar('Loss/duration_loss', d_l, accumulated_current_step)
                    val_logger.add_scalar('Loss/F0_loss', f_l, accumulated_current_step)
                    val_logger.add_scalar('Loss/energy_loss', e_l, accumulated_current_step)
            
                # early stop
                if prev_l <= t_l:
                    early_stop += 1
                    if early_stop == hp.early_stop:
                        print("Early Stop!")
                        return
                    prev_l = t_l
                else: prev_l=0
            
            if accumulated_current_step % hp.save_step == 0:   # 이전보다 val loss가 낮을 때만 저장
                # save best models
                if len(eval_losses) < hp.num_best_model:
                    eval_losses.append((t_l, os.path.join(checkpoint_path, 'checkpoint_{}.pth.tar'.format(accumulated_current_step))))
                    eval_losses.sort(key=lambda x:x[0])
                elif eval_losses[-1][0] > t_l:  # 저장된 모델 중 가장 성능이 안좋은 모델보다 성능이 좋을 때만 저장
                    prev_loss, prev_model = eval_losses.pop()
                    os.system("rm {}".format(prev_model))
                    print('--------------체크포인트 {} 삭제--------------'.format(prev_model))
                else:   # 저장된 모델 중 가장 성능이 안좋은 모델보다 eval loss가 크면 저장 X
                    continue
                
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                )}, os.path.join(checkpoint_path, 'checkpoint_{}.pth.tar'.format(accumulated_current_step)))
                print("save model at step {} ...".format(accumulated_current_step))

                model.train()

            end_time = time.perf_counter()
            Time = np.append(Time, end_time - start_time)
            if len(Time) == hp.clear_Time:
                temp_value = np.mean(Time)
                Time = np.delete(
                    Time, [i for i in range(len(Time))], axis=None)
                Time = np.append(Time, temp_value)
        with open(os.path.join(log_path, "steps.txt"), "a") as step:
                step.write(f"------------epoch {epoch+1}-----------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    args = parser.parse_args()
    print(args)
    main(args)
