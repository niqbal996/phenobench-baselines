import numpy as np
import click
import os
import torch
from os.path import join, dirname, abspath
from torch.utils.data import DataLoader
from dataloaders.datasets import Plants, collate_pdc, Beets
import models
import yaml
from tqdm import tqdm
import cv2


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


    # val_dataset = Plants(datapath=cfg['data']['val'], overfit=cfg['train']['overfit'])
    val_dataset = Beets(datapath=cfg['data']['val'], overfit=cfg['train']['overfit'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['train']['batch_size'], collate_fn=collate_pdc, shuffle=False, drop_last=False, num_workers=cfg['train']['workers'])

    model = models.get_model(cfg)
    weights = torch.load(ckpt_file)["model_state_dict"]
    model.load_state_dict(weights)
    
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out,'plant_bboxes/'), exist_ok=True)

    with torch.autograd.set_detect_anomaly(True):
        model.network.eval()
        for idx, item in enumerate(tqdm(iter(val_loader))):
            with torch.no_grad():

                size = item['image'][0].shape[1]
                predictions = model.test_step(item)

                res_names = item['name']
                for i in range(len(res_names)):
                    fname_box = os.path.join(out,'plant_bboxes',res_names[i].replace('png','txt'))

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
                    img = (item['image'][0] * 255).cpu().numpy().astype(np.uint8)
                    img = cv2.cvtColor(np.transpose(img, (1,2,0)), cv2.COLOR_RGB2BGR)
                    # cx = cx.astype(np.uint8)
                    # cy = cy.astype(np.uint8)
                    # bh = bh.astype(np.uint8)
                    # bw = bw.astype(np.uint8)
                    for idx in range(num_pred):
                        cv2.rectangle(img,
                                      (int(cx[idx] - bw[idx]/2), int(cy[idx] - bh[idx]/2)), 
                                      (int(cx[idx] + bw[idx]/2), int(cy[idx] + bh[idx]/2)),
                                      color=(0, 0, 255),
                                      thickness=2,
                                      )
                        cv2.putText(img,
                                    str(bh[idx]*bw[idx]),
                                      (int(cx[idx] - bw[idx]/2), int(cy[idx] - bh[idx]/2 - 20)),
                                      color=(0, 0, 255),
                                      fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                      fontScale=1,
                                      thickness=2,
                                      )
                    # pred_cls_box_score = np.hstack((labels.reshape(num_pred,1), 
                    #                      cx.reshape(num_pred,1)/size,
                    #                      cy.reshape(num_pred,1)/size,
                    #                      bw.reshape(num_pred,1)/size,
                    #                      bh.reshape(num_pred,1)/size,
                    #                      scores.reshape(num_pred,1)
                    #                     ))
                    # np.savetxt(fname_box, pred_cls_box_score, fmt='%f')
                    cv2.namedWindow('figure')
                    cv2.imshow('figure', img)
                    cv2.waitKey()
                    cv2.destroyWindow('figure')


if __name__ == "__main__":
    main()
