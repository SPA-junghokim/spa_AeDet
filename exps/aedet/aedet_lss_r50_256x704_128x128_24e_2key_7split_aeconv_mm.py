"""
mAP: 0.3594
mATE: 0.6488
mASE: 0.2772
mAOE: 0.4957
mAVE: 0.4318
mAAE: 0.2156
NDS: 0.4728
Eval time: 116.0s

Per-class results:
Object Class    AP  ATE ASE AOE AVE AAE
car 0.525   0.523   0.161   0.129   0.484   0.224
truck   0.289   0.684   0.211   0.124   0.413   0.206
bus 0.416   0.669   0.204   0.098   0.710   0.243
trailer 0.218   0.921   0.231   0.462   0.278   0.198
construction_vehicle    0.089   0.863   0.514   1.200   0.113   0.370
pedestrian  0.307   0.711   0.294   0.897   0.538   0.286
motorcycle  0.374   0.654   0.253   0.587   0.691   0.190
bicycle 0.357   0.521   0.274   0.783   0.226   0.008
traffic_cone    0.485   0.479   0.345   nan nan nan
barrier 0.535   0.462   0.284   0.182   nan nan
"""
from argparse import ArgumentParser, Namespace

import torch
import pytorch_lightning as pl

from callbacks.ema import EMACallback
from exps.aedet.aedet_lss_r50_256x704_128x128_24e_7split import \
    AeDetLightningModel as BaseAeDetLightningModel
from models.aedet_mm import AeDet


class AeDetLightningModel(BaseAeDetLightningModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key_idxes = [-1]
        self.head_conf['bev_backbone_conf']['in_channels'] = 80 * (
            len(self.key_idxes) + 1)
        self.head_conf['bev_neck_conf']['in_channels'] = [
            80 * (len(self.key_idxes) + 1), 160, 320, 640
        ]
        self.model = AeDet(self.backbone_conf,
                              self.head_conf,
                              is_train_depth=True)

    def configure_optimizers(self):
        lr = self.basic_lr_per_img * \
            self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-1)
        return [optimizer]


def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)

    model = AeDetLightningModel(**vars(args))
    print(model)
    train_dataloader = model.train_dataloader()

    tmp = train_dataloader.dataset[1]

    if args.ckpt_path:
        ema_callback = EMACallback(len(train_dataloader.dataset) * args.max_epochs, ema_ckpt_path=args.ckpt_path.replace('origin', 'ema'))
    else:
        ema_callback = EMACallback(len(train_dataloader.dataset) * args.max_epochs)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[ema_callback])
    if args.evaluate:
        trainer.test(model, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model, ckpt_path=args.ckpt_path)


def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('-e',
                               '--evaluate',
                               dest='evaluate',
                               action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-b', '--batch_size_per_device', type=int)
    parent_parser.add_argument('--seed',
                               type=int,
                               default=0,
                               help='seed for initializing training.')
    parent_parser.add_argument('--ckpt_path', type=str)
    parser = AeDetLightningModel.add_model_specific_args(parent_parser)
    parser.set_defaults(profiler='simple',
                        deterministic=False,
                        max_epochs=24,
                        accelerator='ddp',
                        num_sanity_val_steps=0,
                        gradient_clip_val=5,
                        limit_val_batches=1.0,
                        check_val_every_n_epoch=25,
                        enable_checkpointing=False,
                        precision=16,
                        default_root_dir='./outputs/aedet_lss_r50_256x704_128x128_24e_2key_7split_mm')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()
