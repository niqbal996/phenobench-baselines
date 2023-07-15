import numpy as np
import cv2
import click
import os
import torch
from os.path import join, dirname, abspath
from torch.utils.data import DataLoader
from dataloaders.datasets import Leaves, collate_pdc
import models
import yaml


def save_model(model, epoch, optim, name):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        }, name)

@click.command()
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'configs/cfg.yaml'))
@click.option('--ckpt_file',
              '-w',
              type=str,
              help='path to trained weights (.pt)',
              default=join(dirname(abspath(__file__)),'checkpoints/best.pt'))
@click.option('--out',
              '-o',
              type=str,
              help='output directory',
              default=join(dirname(abspath(__file__)),'results/'))
def main(config, ckpt_file, out):
    cfg = yaml.safe_load(open(config))

    val_dataset = Leaves(datapath=cfg['data']['val'], overfit=cfg['train']['overfit'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['train']['batch_size'], collate_fn=collate_pdc, shuffle=False, drop_last=False, num_workers=cfg['train']['workers'])

    model = models.get_model(cfg)
    weights = torch.load(ckpt_file)["model_state_dict"]
    model.load_state_dict(weights)
    
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out,'predictions/leaf_instances/'), exist_ok=True)
    os.makedirs(os.path.join(out,'predictions/semantics/'), exist_ok=True)
    os.makedirs(os.path.join(out,'leaf_bboxes/'), exist_ok=True)

    with torch.autograd.set_detect_anomaly(True):
        model.network.eval()
        for idx, item in enumerate(iter(val_loader)):
            with torch.no_grad():
                size = item['image'][0].shape[1]
                semantic, instance, predictions = model.test_step(item)

                res_names = item['name']
                for i in range(len(res_names)):
                    fname_ins = os.path.join(out,'predictions/leaf_instances/',res_names[i])
                    fname_sem = os.path.join(out,'predictions/semantics/',res_names[i])
                    fname_box = os.path.join(out,'leaf_bboxes',res_names[i].replace('png','txt'))

                    cv2.imwrite(fname_sem, semantic[i].cpu().long().numpy())
                    cv2.imwrite(fname_ins,instance[i].cpu().long().numpy())

                    size = item['image'][i].shape[1]
                    scores = predictions[i]['scores'].cpu().numpy()
                    labels = predictions[i]['labels'].cpu().numpy()
                    # converting boxes to center, width, height format
                    boxes_ = predictions[i]['boxes'].cpu().numpy()
                    num_pred = len(boxes_)
                    cx = (boxes_[:,2] + boxes_[:,0])/2
                    cy = (boxes_[:,3] + boxes_[:,1])/2
                    bw = boxes_[:,2] - boxes_[:,0]
                    bh = boxes_[:,3] - boxes_[:,1]

                    # ready to be saved
                    pred_cls_box_score = np.hstack((labels.reshape(num_pred,1), 
                                         cx.reshape(num_pred,1)/size,
                                         cy.reshape(num_pred,1)/size,
                                         bw.reshape(num_pred,1)/size,
                                         bh.reshape(num_pred,1)/size,
                                         scores.reshape(num_pred,1)
                                        ))
                    np.savetxt(fname_box, pred_cls_box_score, fmt='%f')


if __name__ == "__main__":
    main()
