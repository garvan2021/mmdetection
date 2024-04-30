import argparse
import os

from mmengine import Config

from mmdet.utils import compat_cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Convert old config to new config.')
    parser.add_argument('config', help='original config file path')
    parser.add_argument('save_path', help='save path of modified config')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # modify some fileds to keep the compatibility of config
    cfg = compat_cfg(cfg)

    # save config
    save_path = args.save_path
    if os.path.split(save_path)[0]:
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
    cfg.dump(save_path)
    print(f'Modified config saving at {save_path}')


if __name__ == '__main__':
    main()
