from timm.models import create_model
from timm.data import create_dataset

from mymodels import *
from mydatasets import *

def create_model(args):
    if args.model in mymodels:
        return create_mymodel(args.model)
    else:
        return create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
            drop_path_rate=args.drop_path,
            drop_block_rate=args.drop_block,
            global_pool=args.gp,
            bn_tf=args.bn_tf,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            scriptable=args.torchscript,
            checkpoint_path=args.initial_checkpoint)
    

def create_dataset(args,is_training=True):
    
    if args.dataset in mydatasets:
        return create_mydataset(args.dataset)
    else:
        if is_training:
            return create_dataset(
        args.dataset,
        root=args.data_dir, split=args.train_split, is_training=True,
                batch_size=args.batch_size, repeats=args.epoch_repeats)
        else:
            return create_dataset(
                args.dataset, root=args.data_dir, split=args.val_split, is_training=False, batch_size=args.batch_size)
        
