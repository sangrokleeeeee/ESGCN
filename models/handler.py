import json
from datetime import datetime

from data_loader.forecast_dataloader import ForecastDataset, de_normalized
from models.base_model import WSTGCN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
import numpy as np
import time
import os

from utils.math_utils import evaluate


class Loss(nn.Module):
    def __init__(self, is_mask, normalize_method, statistic, alpha=1):
        super(Loss, self).__init__()
        self.is_mask = is_mask
        self.method = normalize_method
        self.sta = statistic
        self.alpha = alpha

    def denorm(self, pred):
        if self.method == 'z_score':
            return (pred * torch.tensor(self.sta['std']).to(pred.device)) + torch.tensor(self.sta['mean']).to(pred.device)
        else:
            # min max
            min_ = torch.tensor(self.sta['min']).to(pred.device)
            max_ = torch.tensor(self.sta['max']).to(pred.device)
            scale = max_ - min_
            return pred * scale + min_
    
    def masked_mae(self, preds, labels, null_val=np.nan):
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels!=null_val)
        mask = mask.float()
        mask /=  torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(preds-labels)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        # idx = torch.arange(loss.shape[1]).to(loss.device).float()
        # idx = idx.reshape(1, loss.shape[1], 1).exp()
        return loss.mean()

    def forward(self, pred, target, target_unnorm):
        if self.is_mask:
            pred = self.denorm(pred)
            # index = target_unnorm != 0.
            loss = F.smooth_l1_loss(pred, target_unnorm)#)pred[index], target_unnorm[index])
            return loss * self.alpha
            return self.masked_mae(pred, target_unnorm, 0.)
        else:
            # pred = self.denorm(pred)
            return F.l1_loss(pred, target)
            # return F.mse_loss(pred, target_unnorm)


def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)


def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model


def inference(model, dataloader, device, node_cnt, window_size, horizon, save_attention=False):
    forecast_set = []
    target_set = []
    target_unnorm_set = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, target, _, target_unnorm) in enumerate(dataloader):
            inputs = inputs.to(device)
            target = target.to(device)
            # inputs_unnorm = inputs_unnorm.to(device)
            step = 0
            # forecast_result = model(inputs)
            forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float)
            # forecast_steps_unnorm = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float)
            while step < horizon:
                forecast_result = model(inputs)
                # if save_attention:
                #     np.save( f'attention_{step}.npy', a.cpu().numpy())
                len_model_output = forecast_result.shape[1]
                if len_model_output == 0:
                    raise Exception('Get blank inference result')
                inputs[:, :window_size - len_model_output, :] = inputs[:, len_model_output:window_size,
                                                                   :].clone()
                inputs[:, window_size - len_model_output:, :] = forecast_result.clone()
                forecast_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                    forecast_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()
                step += min(horizon - step, len_model_output)
            # forecast_set.append(forecast_result.cpu().numpy())
            forecast_set.append(forecast_steps)
            target_set.append(target.detach().cpu().numpy())
            target_unnorm_set.append(target_unnorm.numpy())
    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0), np.concatenate(target_unnorm_set, axis=0)


def validate(model, dataloader, device, normalize_method, statistic,
             node_cnt, window_size, horizon,
             result_file=None, save_attention=False):
    start = time.time()
    forecast_norm, target_norm, target_unnorm = inference(model, dataloader, device,
                                           node_cnt, window_size, horizon, save_attention)
    
    if normalize_method and statistic:
        forecast = de_normalized(forecast_norm, normalize_method, statistic)
        # target = de_normalized(target_norm, normalize_method, statistic)
        target = target_unnorm
    else:
        forecast, target = forecast_norm, target_norm
    scores = evaluate(target, forecast, dataloader.dataset.is_mask)
    # score_by_node = evaluate(target, forecast,  dataloader.dataset.is_mask, by_node=True)
    end = datetime.now()

    # score_norm = evaluate(target_norm, forecast_norm, dataloader.dataset.is_mask)
    
    if not dataloader.dataset.is_mask:
        print(f'RSE {scores[0]:7.9f}; RAE {scores[1]:7.9f}; CORR {scores[2]:7.9f}')
    else:
        for idx, score in enumerate(scores):
            print(f'{idx+1}: MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}')
    if result_file:
        if not os.path.exists(result_file):
            os.makedirs(result_file)
        step_to_print = 0
        forcasting_2d = forecast[:, step_to_print, :]
        forcasting_2d_target = target[:, step_to_print, :]
        np.savetxt(f'{result_file}/target.csv', forcasting_2d_target, delimiter=",")
        np.savetxt(f'{result_file}/predict.csv', forcasting_2d, delimiter=",")
        np.save(f'{result_file}/target.npy', forecast)
        np.save(f'{result_file}/predict.npy', target)
        np.savetxt(f'{result_file}/predict_abs_error.csv',
                   np.abs(forcasting_2d - forcasting_2d_target), delimiter=",")
        np.savetxt(f'{result_file}/predict_ape.csv',
                   np.abs((forcasting_2d - forcasting_2d_target) / forcasting_2d_target), delimiter=",")
    if not dataloader.dataset.is_mask:
        return dict(mae=scores[1], mape=scores[0],
                rmse=scores[2])
    return dict(mae=score[1], mape=score[0],
                rmse=score[2])


def train(train_data, valid_data, test_data, args, result_file):
    node_cnt = train_data.shape[1]
    
    if len(train_data) == 0:
        raise Exception('Cannot organize enough training data')
    if len(valid_data) == 0:
        raise Exception('Cannot organize enough validation data')

    if args.norm_method == 'z_score':
        print('z_score')
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)
        normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
    elif args.norm_method == 'min_max':
        train_min = np.min(train_data, axis=0)
        train_max = np.max(train_data, axis=0)
        normalize_statistic = {"min": train_min.tolist(), "max": train_max.tolist()}
    else:
        normalize_statistic = None
    if normalize_statistic is not None:
        with open(os.path.join(result_file, 'norm_stat.json'), 'w') as f:
            json.dump(normalize_statistic, f)
    model = WSTGCN(node_cnt, 2, args.window_size, args.multi_layer, args.step_type, normalize_statistic, horizon=args.horizon)
    model.to(args.device)

    if args.optimizer == 'RMSProp':
        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.lr, eps=1e-08, weight_decay=args.weight_decay)
    else:
        my_optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)

    train_set = ForecastDataset(train_data, name=args.dataset, window_size=args.window_size, horizon=args.horizon, step_type=args.step_type,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    valid_set = ForecastDataset(valid_data, name=args.dataset, window_size=args.window_size, horizon=args.horizon, step_type=args.step_type,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic, is_train=False)
    test_set = ForecastDataset(test_data, name=args.dataset, window_size=args.window_size, horizon=args.horizon, step_type=args.step_type,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic, is_train=False)
    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=True, shuffle=True,
                                         num_workers=8)
    valid_loader = torch_data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    forecast_loss = Loss(train_loader.dataset.is_mask, args.norm_method, normalize_statistic, args.alpha)#MaskedMAE() if train_loader.dataset.is_mask else nn.MSELoss(reduction='mean').to(args.device)

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params}")

    best_validate_mae = np.inf
    validate_score_non_decrease_count = 0
    performance_metrics = {}
    iter = 0
    task_level = 1
    # for epoch in range(args.epoch):
    while True:
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        for i, (inputs, target, input_unnorm, target_unnorm) in enumerate(train_loader):
            inputs = inputs.to(args.device)
            target = target.to(args.device)
            input_unnorm = input_unnorm.to(args.device)
            target_unnorm = target_unnorm.to(args.device)
            iter += 1
            model.zero_grad()
            forecast = model(inputs)
            if iter % args.step_cl == 0 and task_level <= args.horizon:
                task_level +=1
            if args.step_type == 'multi' and args.step_cl != -1:
                loss = forecast_loss(forecast[:, :task_level], target[:, :task_level], target_unnorm[:, :task_level])
            else:
                loss = forecast_loss(forecast, target, target_unnorm)
            cnt += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            my_optim.step()
            
            loss_total += float(loss)
            # print(f'step: {i+1}/{len(train_loader)}loss: {loss_total/(i+1)}')
        epoch = 0
        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}'.format(epoch, (
                time.time() - epoch_start_time), loss_total / cnt))
        # forecast = de_normalized(forecast.cpu().detach().numpy(), args.norm_method, normalize_statistic)
        # scores = evaluate(target_unnorm.cpu().detach().numpy(), forecast, train_loader.dataset.is_mask)
        
        # if not train_loader.dataset.is_mask:
        #     print(f'RSE {scores[0]:7.9f}; RAE {scores[1]:7.9f}; CORR {scores[2]:7.9f}')
        # else:
        #     for idx, score in enumerate(scores):
        #         print(f'{idx+1}: MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}')
        # save_model(model, result_file, epoch)
        # if (epoch+1) % args.exponential_decay_step == 0:
        #     my_lr_scheduler.step()
        # if (epoch + 1) % args.validate_freq == 0 and len(valid_data) > 0:
        #     is_best_for_now = False
        #     print('------ validate on data: VALIDATE ------')
        #     performance_metrics = \
        #         validate(model, valid_loader, args.device, args.norm_method, normalize_statistic,
        #                  node_cnt, args.window_size, args.horizon,
        #                  result_file=result_file)
        #     if best_validate_mae > performance_metrics['mae']:
        #         best_validate_mae = performance_metrics['mae']
        #         is_best_for_now = True
        #         validate_score_non_decrease_count = 0
        #     else:
        #         validate_score_non_decrease_count += 1
        #     # save model
        #     if is_best_for_now:
        #         save_model(model, result_file)
                # print('------ validate on data: TEST ------')
                # performance_metrics = \
                #     validate(model, test_loader, args.device, args.norm_method, normalize_statistic,
                #             node_cnt, args.window_size, args.horizon,
                #             result_file=result_file)
        # early stop
        # if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
        #     break
    return performance_metrics, normalize_statistic


def test(test_data, args, result_train_file, result_test_file, save_attention=False):
    with open(os.path.join(result_train_file, 'norm_stat.json'),'r') as f:
        normalize_statistic = json.load(f)
    model = load_model(result_train_file)
    node_cnt = test_data.shape[1]
    test_set = ForecastDataset(test_data, name=args.dataset, window_size=args.window_size, horizon=args.horizon, step_type=args.step_type,
                               normalize_method=args.norm_method, norm_statistic=normalize_statistic, is_train=False)
    test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False,
                                        shuffle=False, num_workers=0)
    performance_metrics = validate(model, test_loader, args.device, args.norm_method, normalize_statistic,
                      node_cnt, args.window_size, args.horizon,
                      result_file=result_test_file, save_attention=save_attention)
    mae, mape, rmse = performance_metrics['mae'], performance_metrics['mape'], performance_metrics['rmse']

    if test_loader.dataset.is_mask:
        with open(os.path.join(result_test_file, 'final.txt'), 'w') as f:
            f.write('Performance on test set: MAPE: {:7.9f} | MAE: {:7.9f} | RMSE: {:7.9f}'.format(mape, mae, rmse))
        print('Performance on test set: MAPE: {:7.9f} | MAE: {:7.9f} | RMSE: {:7.9f}'.format(mape, mae, rmse))
    else:
        with open(os.path.join(result_test_file, 'final.txt'), 'w') as f:
            f.write('Performance on test set: RSE: {:7.9f} | RAE: {:5.4f} | CORR: {:5.4f}'.format(mape, mae, rmse))
        print('Performance on test set: RSE: {:7.9f} | RAE: {:5.4f} | CORR: {:5.4f}'.format(mape, mae, rmse))
