# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist
from torch.utils.data import dataset

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
import torchnet as tnt

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size


logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def save_model_complete(args, model, optimizer, accuracy = None, step = 0):
    if not accuracy:
        checkpoint_file = os.path.join(args.output_dir_every_checkpoint, "step_{}_checkpoint.pth".format(step))
    else:
        checkpoint_file = os.path.join(args.output_dir, "best_acc_step_{}_acc_{}_checkpoint.pth".format(step, accuracy))
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': accuracy,
    }, checkpoint_file)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    if args.dataset == "cifar10":
        num_classes = 10 
    elif args.dataset == "stanford40":
        num_classes = 40
    else:
        num_classes = 100

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    writer.add_scalar("test/loss", scalar_value=eval_losses.avg, global_step=global_step)

    return accuracy

def mAp(args, model, writer, test_loader, global_step):
    if global_step == -1:
        # checkpoint_file = '/content/drive/MyDrive/ViT_acc_90.99/best_acc_step_1100_acc_0.9099783080260304_checkpoint.pth'
        # checkpoint_file = '/content/drive/MyDrive/ViT_layer_11_to_end/best_acc_step_500_acc_0.9063629790310919_checkpoint.pth'
        # checkpoint_file = '/content/best_acc/TrainedModels/best_acc_step_100_acc_0.9054591467823572_checkpoint.pth'
        # checkpoint_file = '/content/best_acc/TrainedModels/best_acc_step_100_acc_0.9070860448300795_checkpoint.pth'
        checkpoint_file = '/content/best_acc/TrainedModels/best_acc_step_100_acc_0.9081706435285611_checkpoint.pth'
        checkpoint_continue = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint_continue['model_state_dict'], strict=True)

    model.eval()
    class_acc = tnt.meter.APMeter()
    test_map = tnt.meter.mAPMeter()
    topacc = tnt.meter.ClassErrorMeter(topk=[1, 5], accuracy=False)
    conf_matrix = tnt.meter.ConfusionMeter(k=40,  normalized =False)
    class_acc.reset()
    test_map.reset()
    conf_matrix.reset()
    topacc.reset()

    with torch.no_grad():
        for x2, y in test_loader:

            x2 = x2.to(args.device)
            y = y.to(args.device)
            one_hot_y = torch.nn.functional.one_hot(y, num_classes=40)
            one_hot_y = one_hot_y.to(args.device)
            outputs = model(x2)[0]
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            class_acc.add(probs, one_hot_y)
            test_map.add(probs, one_hot_y)
            conf_matrix.add(probs, one_hot_y)
            topacc.add(probs, y)
            
    logger.info('class accs are {}'.format(class_acc.value()))
    logger.info('mAp is equal to {}'.format(test_map.value()))
    logger.info('confusion matrix is {}'.format(conf_matrix.value()))
    logger.info('top 1th and 5th acc values are {}'.format(topacc.value()))

    writer.add_scalar("test/mAp", scalar_value=test_map.value(), global_step=global_step)

def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)


  #  # Trainable Parameters
  #   for name, param in model.named_parameters():
  #       if 'transformer.encoder.layer.11' in name:
  #           param.requires_grad_(True)
  #           print(name)
  #       elif 'transformer.encoder.layer.10' in name:
  #           param.requires_grad_(True)
  #           print(name)
  #       elif 'head.weight' in name or \
  #           'head.bias' in name:
  #           param.requires_grad_(True)
  #           print(name)
  #       elif 'transformer.encoder.encoder_norm.weight' in name \
  #               or 'transformer.encoder.encoder_norm.bias' in name:
  #           param.requires_grad_(True)
  #           print(name)
  #       else:
  #          param.requires_grad_(False)


    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(),
    #                             lr=args.learning_rate,
    #                             # momentum=0.9,
    #                             # weight_decay=args.weight_decay
    #                             )
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= t_total, eta_min=0, last_epoch=-1, verbose=False)

    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    global_step, best_acc = 0, 0

    # load weights
    # checkpoint_file = os.listdir(args.input_dir)
    if args.input_dir:
        checkpoint_file = args.input_dir
        # if checkpoints:
            # checkpoint_file = os.path.join(args.input_dir, checkpoints[0])
        checkpoint = torch.load(checkpoint_file)
        if 'model_state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # global_step = checkpoint['step'] + 1
            best_acc = checkpoint['best_accuracy']
        else:
            model.load_state_dict(checkpoint)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()

    mAp(args, model, writer, test_loader, global_step = -1)

    # global_step, best_acc = 0, 0
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            loss = model(x, y)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)
                # save_checkpoint
                # save_model_complete(args, model, optimizer, accuracy = None, step = global_step)
                
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy = valid(args, model, writer, test_loader, global_step)
                    # writer.add_scalar("test/acc", scalar_value=accuracy, global_step=global_step)
                    mAp(args, model, writer, test_loader, global_step)

                    if best_acc < accuracy:
                        save_model_complete(args, model, optimizer, accuracy, global_step)
                        best_acc = accuracy
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "stanford40"], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="/content/best_acc/TrainedModels/", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--output_dir_every_checkpoint", 
                        default="/content/every_checkpoint/TrainedModels/", type=str,
                        help="The output directory where checkpoints will be written.")
    # parser.add_argument("--input_dir", 
    #                     default="/content/drive/MyDrive/ViT_weights_layer11_to_end/best_acc_step_500_acc_0.9063629790310919_checkpoint.pth", type=str,
    #                     help="The output directory where checkpoints will be written.")
    parser.add_argument("--input_dir", 
                        default= None, type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    # parser.add_argument("--train_batch_size", default=40, type=int,
    #                     help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    # parser.add_argument("--learning_rate", default=3e-4, type=float,
    #                     help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    

    # Training
    train(args, model)


if __name__ == "__main__":
    main()

   
