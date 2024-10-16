import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='command for evaluate on CUHK-PEDES')

    # Directory
    #CUHK-PEDES
    parser.add_argument('--dir', type=str,
                        default=r'/datasets',
                        help='directory to store dataset')
    parser.add_argument('--dataset', type=str,
                        default="CUHKPEDES")

    #image setting
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--height', type=int, default=384)

    # CNN setting
    parser.add_argument('--num_classes', type=int, default=11003)
    parser.add_argument('--pretrained', action='store_true',
                        help='whether or not to restore the pretrained visual model')
    parser.add_argument('--droprate', default=0, type=float, help='drop rate')

    # CLIP setting
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--embedding_type', type=str,
                        default='CLIP')


    parser.add_argument('--feature_size', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epoches', type=int, default=80)
    parser.add_argument('--gpus', type=str, default='4')

    args = parser.parse_args()
    return args
