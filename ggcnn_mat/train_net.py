import datetime
import os
import sys
import argparse
import logging

import cv2
import time
import numpy as np

import random
import torch
import torch.utils.data
import torch.optim as optim

import tensorboardX
from torchsummary import summary

from utils.dataset_processing.evaluation import evaluation
from utils.saver import Saver
from models import get_network
from models.common import post_process_output
from models.loss import focal_loss
from utils.data.grasp_data import GraspDataset
from utils.visualisation.gridshow import gridshow

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Train GG-CNN')

    # Network
    parser.add_argument('--network', type=str, default='ggcnn', help='Network Name in .models')

    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, default='cornell', help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, default='E:/dataset/cornell', help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for training (0/1)')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=4, help='Dataset workers')

    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=1000, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=250, help='Validation Batches')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0, help='权重衰减 L2正则化系数')
    parser.add_argument('--output-size', type=int, default=300, help='output size')

    # Logging etc.
    parser.add_argument('--description', type=str, default='', help='Training description')
    parser.add_argument('--modeldir', type=str, default='models', help='model保存地址')
    parser.add_argument('--outdir', type=str, default='output', help='Training Output Directory')
    parser.add_argument('--logdir', type=str, default='tensorboard', help='Log directory')
    parser.add_argument('--imgdir', type=str, default='img', help='中间预测图保存文件夹')
    parser.add_argument('--vis', action='store_true', help='Visualise the training process')

    # 从已有网络继续训练
    parser.add_argument('--goon-train', type=bool, default=False, help='是否从已有网络继续训练')
    parser.add_argument('--model', type=str, default='output/models/220728_0920_redwallbot/epoch_0081_acc_0.2838.pth',
                        help='保存的模型')
    parser.add_argument('--start-epoch', type=int, default=82, help='继续训练开始的epoch')

    args = parser.parse_args()
    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def validate(net, device, val_data, batches_per_epoch):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param batches_per_epoch: Number of batches to run
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'accuracy': 0.0,
        'graspable': 0,
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {
        }
    }

    ld = len(val_data)

    with torch.no_grad():
        batch_idx = 0
        while batch_idx < batches_per_epoch:
            for x, y in val_data:
                batch_idx += 1
                if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
                    break

                xc = x.to(device)
                yc = [yy.to(device) for yy in y]
                # 预测并计算损失
                lossd = focal_loss(net, xc, yc[0], yc[1], yc[2], yc[3])
                # lossd = net.compute_loss(xc, yc)

                loss = lossd['loss']

                results['loss'] += loss.item() / ld
                for ln, l in lossd['losses'].items():
                    if ln not in results['losses']:
                        results['losses'][ln] = 0
                    results['losses'][ln] += l.item() / ld

                # q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                #                                             lossd['pred']['sin'], lossd['pred']['width'])
                # 输出值预处理
                pos_out, ang_out, wid_out = post_process_output(lossd['pred']['pred_pos'],
                                                                lossd['pred']['pred_cos'],
                                                                lossd['pred']['pred_sin'],
                                                                lossd['pred']['pred_wid'])
                results['graspable'] += np.max(pos_out) / ld

                # 评估
                ang_tar = torch.atan2(y[2], y[1]) / 2.0
                ret = evaluation(pos_out, ang_out, wid_out, y[0], ang_tar, y[3])
                results['accuracy'] += ret / ld
                # s = evaluation.calculate_iou_match.calculate_iou_match(q_out, ang_out,
                #                                    val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                #                                    no_grasps=1,
                #                                    grasp_width=w_out,
                #                                    )

                if ret:
                    results['correct'] += 1
                else:
                    results['failed'] += 1

    return results


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    while batch_idx < batches_per_epoch:
        for x, y in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break

            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            # lossd = net.compute_loss(xc, yc)
            # 计算损失
            lossd = focal_loss(net, xc, yc[0], yc[1], yc[2], yc[3])

            loss = lossd['loss']

            if batch_idx % 100 == 0:
                # logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))
                logging.info('Epoch: {}, '
                             'Batch: {}, '
                             'loss_pos: {:.5f}, '
                             'loss_cos: {:.5f}, '
                             'loss_sin: {:.5f}, '
                             'loss_wid: {:.5f}, '
                             'Loss: {:0.5f}'.format(epoch, batch_idx, lossd['losses']['loss_pos'],
                                                    lossd['losses']['loss_cos'], lossd['losses']['loss_sin'],
                                                    lossd['losses']['loss_wid'], loss.item()))

            # 统计损失
            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Display the images
            if vis:
                imgs = []
                n_img = min(4, x.shape[0])
                for idx in range(n_img):
                    imgs.extend([x[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
                        x[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in
                                                      lossd['pred'].values()])
                gridshow('Display', imgs,
                         [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0),
                          (0.0, 1.0)] * 2 * n_img,
                         [cv2.COLORMAP_BONE] * 10 * n_img, 10)
                cv2.waitKey(2)

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results


def datasetloaders(Dataset, args):
    # 训练集
    train_dataset = Dataset(args.dataset_path,
                            start=0.0,
                            end=0.9,
                            output_size=args.output_size,
                            argument=True)
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    # 部分训练集->验证
    train_val_dataset = Dataset(args.dataset_path,
                                start=0.3,
                                end=0.5,
                                output_size=args.output_size,
                                argument=False)
    train_val_data = torch.utils.data.DataLoader(
        train_val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1)

    # 测试集
    val_dataset = Dataset(args.dataset_path,
                          start=0.9,
                          end=1.0,
                          output_size=args.output_size,
                          argument=True)
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1)

    return train_data, train_val_data, val_data


def run():
    # 设置随机数种子
    setup_seed(42)
    args = parse_args()

    # 设置保存器
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))
    saver = Saver(args.outdir, args.logdir, args.modeldir, args.imgdir, net_desc)
    # 初始化tensorboard 保存器
    tb = saver.save_summary()
    model_path = os.path.join(args.outdir, args.modeldir, net_desc)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # 加载数据集
    logging.info('Loading Dataset...')
    train_data, train_val_data, val_data = datasetloaders(GraspDataset, args)
    print('>> train dataset: {}'.format(len(train_data) * args.batch_size))
    print('>> train_val dataset: {}'.format(len(train_val_data)))
    print('>> test dataset: {}'.format(len(val_data)))

    # 加载网络
    logging.info('Loading Network...')
    ggcnn = get_network(args.network)
    net = ggcnn()
    device_name = args.device_name if torch.cuda.is_available() else "cpu"
    if args.goon_train:
        # 加载预训练模型
        pretrained_dict = torch.load(args.model, map_location=torch.device(device_name))
        net.load_state_dict(pretrained_dict, strict=True)  # True:完全吻合，False:只加载键值相同的参数，其他加载默认值。
    device = torch.device(device_name)  # 指定运行设备
    net = net.to(device)

    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.5)  # 学习率衰减    20, 30, 60
    logging.info('optimizer Done')

    # 打印网络结构
    summary(net, (1, args.output_size, args.output_size))  # 将网络结构信息输出到终端
    saver.save_arch(net, (1, args.output_size, args.output_size))  # 保存至文件 output/arch.txt

    # 训练
    best_acc = 0.0
    start_epoch = args.start_epoch if args.goon_train else 0
    for _ in range(start_epoch):
        scheduler.step()
    for epoch in range(args.epochs)[start_epoch:]:
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        epoch_start_time = time.time()
        logging.info('Epoch {:02d}, lr={}'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))

        # 训练
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch, vis=args.vis)
        scheduler.step()

        # 保存训练日志
        tb.add_scalar('train_loss/loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        # Run Validation
        # ====================== 使用测试集验证 ======================
        logging.info('Validating...')
        test_results = validate(net, device, val_data, args.val_batches)
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct'] / (test_results['correct'] + test_results['failed'])))
        # 打印日志
        logging.info('>>> test_graspable = {:.5f}'.format(test_results['graspable']))
        logging.info('>>> test_accuracy: %f' % (test_results['accuracy']))

        # 保存测试集日志
        tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
        tb.add_scalar('test_pred/test_graspable', test_results['graspable'], epoch)
        tb.add_scalar('test_pred/test_accuracy', test_results['accuracy'], epoch)
        tb.add_scalar('test_loss/loss', test_results['loss'], epoch)
        for n, l in test_results['losses'].items():
            tb.add_scalar('test_loss/' + n, l, epoch)

        # ====================== 使用部分训练集进行验证 ======================
        train_val_results = validate(net, device, train_val_data, args.val_batches)

        print('\n>>> train_val_graspable = {:.5f}'.format(train_val_results['graspable']))
        print('>>> train_val_accuracy: %f' % (train_val_results['accuracy']))

        tb.add_scalar('train_val_pred/train_val_graspable', train_val_results['graspable'], epoch)
        tb.add_scalar('train_val_pred/train_val_accuracy', train_val_results['accuracy'], epoch)
        tb.add_scalar('train_val_loss/loss', train_val_results['loss'], epoch)
        for n, l in train_val_results['losses'].items():
            tb.add_scalar('train_val_loss/' + n, l, epoch)

        # 保存模型
        accuracy = test_results['accuracy']
        if accuracy >= best_acc or epoch % 10 == 0:
            print('>>> save model: ', 'epoch_%04d_acc_%0.4f.pth' % (epoch, accuracy))
            torch.save(net, os.path.join(model_path, 'epoch_%02d_iou_%0.2f' % (epoch, accuracy)))
            torch.save(net.state_dict(), os.path.join(model_path, 'epoch_%02d_iou_%0.2f_statedict.pt' % (epoch, accuracy)))
            best_acc = accuracy

        epoch_end_time = time.time()
        logging.info('Epoch {:02d} spend {:02f} s'.format(epoch, (epoch_end_time - epoch_start_time)))

    tb.close()


if __name__ == '__main__':
    run()
