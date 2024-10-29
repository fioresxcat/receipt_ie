import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import yaml
from model import BaselineModel, BaselineMLPModel
from dataset_baseline import BaselineGraphDataModule
from my_utils import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from label_list_hub import all_label_list
import shutil
import pdb
from infer_baseline import get_input_from_json, inference


def train(general_cfg, model_cfg):
    experiment_dir = os.path.join(general_cfg['training']['ckpt_save_dir'], general_cfg['training']['exp_description'])
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'train_cfg.yaml'), 'w') as f:
        yaml.dump(general_cfg, f)
    with open(os.path.join(experiment_dir, 'model_cfg.yaml'), 'w') as f:
        yaml.dump(model_cfg, f)

    # get data
    data_module = BaselineGraphDataModule(config=general_cfg)
    data_module.train_ds.save_data_info(out_path=os.path.join(experiment_dir, 'train_data_info.txt'))
    data_module.val_ds.save_data_info(out_path=os.path.join(experiment_dir, 'val_data_info.txt'))
    
    # init model
    label_list = all_label_list[general_cfg['data']['martname']]
    model = load_model(
        general_cfg, 
        model_cfg, 
        n_classes=len(label_list), 
        ckpt_path=None
    )
    print('MODEL TYPE: ', type(model))
    print('DATASET TYPE: ', type(data_module))
    pdb.set_trace()

    # callbacks
    model_ckpt = ModelCheckpoint(
        monitor='val_f1',
        mode='max',
        dirpath=experiment_dir,
        filename='model-{epoch:02d}-{train_loss:.3f}-{val_loss:.3f}-{val_f1:.3f}',
        save_top_k=1,
        auto_insert_metric_name=True,
        every_n_epochs=1
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stop = EarlyStopping(
        monitor="val_f1", 
        mode="max",
        stopping_threshold=1,
        patience=20,
    )
    rich_bar = RichProgressBar(leave=True)

    # tensorboard logger
    logger = TensorBoardLogger(
        save_dir=experiment_dir,
        name='',
        version=''
    )
    
    # trainer
    trainer = Trainer(
        accelerator='cuda',
        # devices="1",
        max_epochs=general_cfg['training']['num_epoch'],
        # max_epochs=10,
        # auto_scale_batch_size=False,
        callbacks=[model_ckpt, lr_monitor, early_stop, rich_bar],
        logger=logger,
        log_every_n_steps=1,
        # overfit_batches=1,
        # fast_dev_run=True,
    )

    # train
    if general_cfg['training']['prev_ckpt_path'] is not None:
        trainer.fit(model=model, datamodule=data_module, ckpt_path=general_cfg['training']['prev_ckpt_path'])
    else:
        trainer.fit(model=model, datamodule=data_module)

    # load best model and infer
    best_ckpt_path = trainer.checkpoint_callback.state_dict()['best_model_path']
    inference(
        ckpt_path=best_ckpt_path,
        src_dir=general_cfg['data']['val_dir'],
        out_dir=os.path.join('model_output', general_cfg['data']['martname'], Path(experiment_dir).name, Path(general_cfg['data']['val_dir']).name)
    )
    inference(
        ckpt_path=best_ckpt_path,
        src_dir=general_cfg['data']['test_dir'],
        out_dir=os.path.join('model_output', general_cfg['data']['martname'], Path(experiment_dir).name, Path(general_cfg['data']['test_dir']).name)
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some arguments...')
    parser.add_argument('--mart_name', type=str, required=True)
    args = parser.parse_args()

    martname = args.mart_name

    # load config and setup
    with open("configs/train_baseline_cfg.yaml") as f:
        general_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    model_type = general_cfg['options']['model_type']
    with open(os.path.join('configs', model_type+'.yaml')) as f:
        model_cfg = yaml.load(f, Loader=yaml.FullLoader)

    # modify to mart type
    general_cfg['data']['martname'] = martname
    general_cfg['data']['train_dir'] = f'data/{martname}/train'
    general_cfg['data']['val_dir'] = f'data/{martname}/val'
    general_cfg['data']['test_dir'] = f'data/{martname}/test'
    general_cfg['training']['ckpt_save_dir'] = f'ckpt/{martname}'

    # for martname, data in list(DATA_DICT.items())[18:]:
    #     print(f'---------------------- Training for {martname} ---------------------------')
    #     general_cfg['data']['train_dir'] = data['train_dir']
    #     general_cfg['data']['val_dir'] = data['val_dir']
    #     general_cfg['data']['test_dir'] = data['test_dir']
    #     general_cfg['data']['martname'] = martname
    #     general_cfg['data']['label_list'] = martname
    #     general_cfg['training']['ckpt_save_dir'] = f'hieu_ckpt5/{martname}/'
    #     train(general_cfg, model_cfg)


        # try:
        #     train(general_cfg, model_cfg)
        # except Exception as e:
        #     print(e)
        #     continue

    train(general_cfg, model_cfg)

