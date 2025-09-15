from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from data_provider.data_loader_anomaly import get_anomaly_loader
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'anomaly': 'anomaly',  # Special key for anomaly detection
}


def data_provider(args, flag):
    # Special handling for anomaly detection
    if args.data == 'anomaly':
        # Use os.path.join for proper path construction
        import os
        data_path = os.path.join(args.root_path, args.data_path)
        
        # Use anomaly data loader
        data_set, data_loader = get_anomaly_loader(
            data_path=data_path,
            batch_size=args.batch_size if flag != 'pred' else 1,
            win_size=args.seq_len,
            step=args.stride if hasattr(args, 'stride') else 1,
            mode=flag,
            num_workers=args.num_workers
        )
        
        print(flag, len(data_set))
        return data_set, data_loader
    
    # Standard Autoformer data loading for other datasets
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
