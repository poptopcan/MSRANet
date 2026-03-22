from yacs.config import CfgNode as CN

dataset_cfg = CN()

root = "E:\dataset"
# config for dataset
dataset_cfg.sysu = CN()
dataset_cfg.sysu.num_id = 395
dataset_cfg.sysu.num_cam = 6
dataset_cfg.sysu.data_root = root + '\SYSU-MM01'

dataset_cfg.regdb = CN()
dataset_cfg.regdb.num_id = 206
dataset_cfg.regdb.num_cam = 2
dataset_cfg.regdb.data_root = root + '\RegDB'

dataset_cfg.llcm = CN()
dataset_cfg.llcm.num_id = 713
dataset_cfg.llcm.num_cam = 9
dataset_cfg.llcm.data_root = root + '\LLCM'