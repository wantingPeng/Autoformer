import os
import torch
import numpy as np
import argparse
from exp.exp_main import Exp_Main
from utils.logger import logger
import random

def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    # 创建解析器
    parser = argparse.ArgumentParser(description='Autoformer for Anomaly Detection')
    
    # 基础配置
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='anomaly_test', help='model id')
    parser.add_argument('--model', type=str, default='Autoformer', help='model name')
    
    # 数据加载器
    parser.add_argument('--data', type=str, default='anomaly', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ALLcontact_noSegment/', help='data file')
    parser.add_argument('--features', type=str, default='M', help='features')
    parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    
    # 序列长度
    parser.add_argument('--seq_len', type=int, default=10, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=0, help='prediction sequence length')
    parser.add_argument('--stride', type=int, default=1, help='step size for sliding window')
    
    # 模型定义
    parser.add_argument('--enc_in', type=int, default=27, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=27, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=27, help='output size')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false', default=True, help='whether to use distilling in encoder')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    
    # 优化
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', default=False, help='use automatic mixed precision training')
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multiple gpus')
    
    args = parser.parse_args()
    
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    logger.info('Args in experiment:')
    logger.info(args)
    
    # 设置实验
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des)
    
    # 创建实验
    exp = Exp_Main(args)
    
    # 训练模型
    logger.info('>>>>>>>开始训练 : {}>>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    try:
        exp.train(setting)
    except Exception as e:
        logger.error(f"训练时发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
    # 测试模型
    logger.info('>>>>>>>测试 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    try:
        exp.test(setting)
    except Exception as e:
        logger.error(f"测试时发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
