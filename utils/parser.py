import argparse


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeError('Boolean value expected.')
    

def parse_args():
    parser = argparse.ArgumentParser()

    general_group = parser.add_argument_group('general_group')
    general_group.add_argument('--config', type=str, required=True,
                               help='Specify one of the configuration files in ./config.')
    general_group.add_argument('--save_dir', type=str, default=argparse.SUPPRESS,
                                help='Will be determined automatically if not otherwise specified.')
    general_group.add_argument('--top_dir', type=str, default='',
                               help='save_dir = top_dir/auto_save_dir. ignored when save_dir is specified.')
    general_group.add_argument('--debug', action='store_true', 
                               help='do a mini-test using minimum schedule.')
    general_group.add_argument('--no_pbar', action='store_true', 
                               help='Suppress printing pregress bars.')
    general_group.add_argument('--no_color', action='store_true', 
                               help='Suppress colored text(if not supported).')
    general_group.add_argument('--overwrite', action='store_true', 
                               help='Reuse/ovewrite the latest/given save_dir.')
    general_group.add_argument('--fake_checkpoint', action='store_true', 
                               help='Save fake checkpoints for memory efficient debugging.')
    general_group.add_argument('--log_level', type=str, default='debug')
    general_group.add_argument('--detect_anomaly', action='store_true')
    general_group.add_argument('--disable_autocast', action='store_true')

    multinode_group = parser.add_argument_group('multinode_group')
    multinode_group.add_argument('--dist_url', type=str, default='local')
    multinode_group.add_argument('--num_machines', type=int, default=1)
    multinode_group.add_argument('--machine_rank', type=int, default=0)
    multinode_group.add_argument('--num_gpus', type=int, default=0)

    overwrite_group = parser.add_argument_group('configs that take precedence over .yaml')
    """NOTE:
    The arguments in this group have the same name as the fields in the YAML files, 
    but they must have the delimiter '/' to indicate the hierarchy.
    If the user specify those arguments, the correponding values in the YAML file will be overwritten. 
    """
    overwrite_group.add_argument('--load_dir', type=str, default=argparse.SUPPRESS, 
                                 help='Set the load_dir different from save_dir. (default: save_dir = load_dir)')
    overwrite_group.add_argument('--augment/type', type=str, default=argparse.SUPPRESS)
    overwrite_group.add_argument('--network/name', type=str, default=argparse.SUPPRESS)
    overwrite_group.add_argument('--train/enabled', type=str2bool, default=argparse.SUPPRESS)
    overwrite_group.add_argument('--train/num_workers', type=int, default=argparse.SUPPRESS)
    overwrite_group.add_argument('--train/max_epochs', type=int, default=argparse.SUPPRESS)
    overwrite_group.add_argument('--train/batch_size_train', type=int, default=argparse.SUPPRESS)
    overwrite_group.add_argument('--train/batch_size_eval', type=int, default=argparse.SUPPRESS)
    overwrite_group.add_argument('--train/optim/lr', type=float, default=argparse.SUPPRESS)
    overwrite_group.add_argument('--train/optim/weight_decay', type=float, default=argparse.SUPPRESS)
    overwrite_group.add_argument('--eval/enabled', type=str2bool, default=argparse.SUPPRESS)
    overwrite_group.add_argument('--eval/mode', type=str, default=argparse.SUPPRESS)
    overwrite_group.add_argument('--eval/dataset', type=str, default=argparse.SUPPRESS)
    overwrite_group.add_argument('--eval/num_workers', type=int, default=argparse.SUPPRESS)
    overwrite_group.add_argument('--eval/batch_size', type=int, default=argparse.SUPPRESS)
    overwrite_group.add_argument('--eval/optim/lr', type=float, default=argparse.SUPPRESS)

    return parser.parse_args()
