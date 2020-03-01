import os
from argparse import ArgumentParser
from utils import DefaultBoxes, Encoder, COCODetection
from base_model import Loss
from utils import SSDTransformer
from ssd300 import SSD300
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils import mkldnn as mkldnn_utils
import time
import random
import numpy as np
from mlperf_compliance import mlperf_log
from mlperf_logger import ssd_print, broadcast_seeds

def parse_args():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")
    parser.add_argument('--data', '-d', type=str, default='/coco',
                        help='path to test and training data files')
    parser.add_argument('--epochs', '-e', type=int, default=800,
                        help='number of epochs for training')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                        help='number of examples for each iteration')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int, default=random.SystemRandom().randint(0, 2**32 - 1),
                        help='manually set random seed for torch')
    parser.add_argument('--threshold', '-t', type=float, default=0.23,
                        help='stop training early at threshold')
    parser.add_argument('--iteration', type=int, default=0,
                        help='iteration to start from')
    parser.add_argument('--totle-iteration', type=int, default=0,
                        help='iteration to run performance test, 0 means no limited')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to model checkpoint file')
    parser.add_argument('--no-save', action='store_true',
                        help='save model checkpoints')
    parser.add_argument('--evaluation', nargs='*', type=int,
                        default=[40, 50, 55, 60, 65, 70, 75, 80],
                        help='epochs at which to evaluate')
    parser.add_argument('-m', '--mode', metavar='MODE', default='training',
                    choices=["training", "inference"],
                    help='running mode: ' +
                        ' | '.join(["training", "inference"]) +
                        ' (default: training)')
    parser.add_argument('--lr-decay-schedule', nargs='*', type=int,
                        default=[40, 50],
                        help='epochs at which to decay the learning rate')
    parser.add_argument('--warmup', type=float, default=None,
                        help='how long the learning rate will be warmed up in fraction of epochs')
    parser.add_argument('--warmup-factor', type=int, default=0,
                        help='mlperf rule parameter for controlling warmup curve')
    parser.add_argument('--perf-prerun-warmup', type=int, default=0,
                        help='how much iterations to pre run before performance test, -1 mean use all dataset.')
    parser.add_argument('--lr', type=float, default=2.5e-3,
                        help='base learning rate')
    # Distributed stuff
    parser.add_argument('--local_rank', default=0, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                        'or automatically set by using \'python -m multiproc\'.')
    parser.add_argument('--use-mkldnn', action='store_true',
                        help='use mkldnn')

    return parser.parse_args()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def show_memusage(device=0):
    import gpustat
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}/{}".format(item["memory.used"], item["memory.total"]))


def dboxes300_coco():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    ssd_print(key=mlperf_log.FEATURE_SIZES, value=feat_size)

    steps = [8, 16, 32, 64, 100, 300]
    ssd_print(key=mlperf_log.STEPS, value=steps)

    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]
    ssd_print(key=mlperf_log.SCALES, value=scales)

    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    ssd_print(key=mlperf_log.ASPECT_RATIOS, value=aspect_ratios)

    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    ssd_print(key=mlperf_log.NUM_DEFAULTS,
                         value=len(dboxes.default_boxes))
    return dboxes


def coco_eval(model, coco, cocoGt, encoder, inv_map, threshold,
              epoch, iteration, use_cuda=True, is_inference_test=False, warmup=0, use_mkldnn=False):
    from pycocotools.cocoeval import COCOeval

    batch_time = AverageMeter('Time', ':6.3f')

    print("")
    model.eval()
    if use_cuda:
        model.cuda()
    elif use_mkldnn and is_inference_test:
        print("using mkldnn model to do inference\n")
        model = mkldnn_utils.to_mkldnn(model)

    ret = []

    overlap_threshold = 0.50
    nms_max_detections = 200
    ssd_print(key=mlperf_log.NMS_THRESHOLD,
                         value=overlap_threshold, sync=False)
    ssd_print(key=mlperf_log.NMS_MAX_DETECTIONS,
                         value=nms_max_detections, sync=False)

    ssd_print(key=mlperf_log.EVAL_START, value=epoch, sync=False)

    totle_num_imgs = len(coco)
    assert totle_num_imgs >= warmup
    start = time.time()
    for idx, image_id in enumerate(coco.img_keys):
        img, (htot, wtot), _, _ = coco[idx]

        with torch.no_grad():
            print("Parsing image: {}/{}".format(idx+1, len(coco)))
            inp = img.unsqueeze(0)
            if use_cuda:
                inp = inp.cuda()
            elif use_mkldnn:
                inp = inp.to_mkldnn()

            if warmup > 0 and idx >= warmup:
                start_time=time.time()

            ploc, plabel = model(inp)

            try:
                result = encoder.decode_batch(ploc, plabel,
                                              overlap_threshold,
                                              nms_max_detections)[0]

            except:
                #raise
                print("")
                print("No object detected in idx: {}".format(idx))
                continue
            finally:
                if warmup > 0 and idx >= warmup:
                    batch_time.update(time.time()-start_time)

            loc, label, prob = [r.cpu().numpy() for r in result]
            for loc_, label_, prob_ in zip(loc, label, prob):
                ret.append([image_id, loc_[0]*wtot, \
                                      loc_[1]*htot,
                                      (loc_[2] - loc_[0])*wtot,
                                      (loc_[3] - loc_[1])*htot,
                                      prob_,
                                      inv_map[label_]])
    print("")
    if is_inference_test:
        latency = batch_time.sum / totle_num_imgs * 1000
        perf = totle_num_imgs / batch_time.sum
        print('inference latency %3.0f ms'%latency)
        print('inference performance %3.0f fps'%perf)
        return True
    else:
        print("Predicting Ended, total time: {:.2f} s".format(time.time()-start))

    cocoDt = cocoGt.loadRes(np.array(ret))

    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.evaluate()
    E.accumulate()
    E.summarize()
    print("Current AP: {:.5f} AP goal: {:.5f}".format(E.stats[0], threshold))

    # put your model back into training mode
    model.train()

    current_accuracy = E.stats[0]
    ssd_print(key=mlperf_log.EVAL_SIZE, value=idx + 1, sync=False)
    ssd_print(key=mlperf_log.EVAL_ACCURACY,
                         value={"epoch": epoch,
                                "value": current_accuracy},
              sync=False)
    ssd_print(key=mlperf_log.EVAL_ITERATION_ACCURACY,
                         value={"iteration": iteration,
                                "value": current_accuracy},
              sync=False)
    ssd_print(key=mlperf_log.EVAL_TARGET, value=threshold, sync=False)
    ssd_print(key=mlperf_log.EVAL_STOP, value=epoch, sync=False)
    return current_accuracy>= threshold #Average Precision  (AP) @[ IoU=050:0.95 | area=   all | maxDets=100 ]


def eval300_mlperf_coco(args):
    global torch
    from coco import COCO
    # Check that GPUs are actually available
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    input_size = 300
    val_trans = SSDTransformer(dboxes, (input_size, input_size), val=True)

    val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.data, "val2017")

    cocoGt = COCO(annotation_file=val_annotate)
    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    inv_map = {v:k for k,v in val_coco.label_map.items()}

    ssd300 = SSD300(val_coco.labelnum)

    if args.checkpoint is not None:
        print("loading model checkpoint", args.checkpoint)
        od = torch.load(args.checkpoint)
        ssd300.load_state_dict(od["model"])

    # if use_cuda:
    #     ssd300.cuda()
    # elif args.use_mkldnn:
    #     logger.info("using mkldnn model to do inference\n")
    #     ssd300 = mkldnn_utils.to_mkldnn(ssd300)

    loss_func = Loss(dboxes)
    if use_cuda:
        loss_func.cuda()

    return coco_eval(ssd300, val_coco, cocoGt, encoder, inv_map, args.threshold, 0, 0, use_cuda=use_cuda, is_inference_test=True, warmup=args.perf_prerun_warmup, use_mkldnn=args.use_mkldnn)


def lr_warmup(optim, wb, iter_num, base_lr, args):
    if iter_num < wb:
        # mlperf warmup rule
        warmup_step = base_lr / (wb * (2 ** args.warmup_factor))
        new_lr = base_lr - (wb - iter_num) * warmup_step
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr


def train300_mlperf_coco(args):
    global torch
    from coco import COCO

    batch_time = AverageMeter('Time', ':6.3f')

    # Check that GPUs are actually available
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.distributed = False
    if use_cuda:
        try:
            from apex.parallel import DistributedDataParallel as DDP
            if 'WORLD_SIZE' in os.environ:
                args.distributed = int(os.environ['WORLD_SIZE']) > 1
        except:
            raise ImportError("Please install APEX from https://github.com/nvidia/apex")

    if args.distributed:
        # necessary pytorch imports
        import torch.utils.data.distributed
        import torch.distributed as dist
 #     ssd_print(key=mlperf_log.RUN_SET_RANDOM_SEED)
        if args.no_cuda:
            device = torch.device('cpu')
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device('cuda')
            dist.init_process_group(backend='nccl',
                                    init_method='env://')
            # set seeds properly
            args.seed = broadcast_seeds(args.seed, device)
            local_seed = (args.seed + dist.get_rank()) % 2**32
            print(dist.get_rank(), "Using seed = {}".format(local_seed))
            torch.manual_seed(local_seed)
            np.random.seed(seed=local_seed)


    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)

    input_size = 300
    train_trans = SSDTransformer(dboxes, (input_size, input_size), val=False)
    val_trans = SSDTransformer(dboxes, (input_size, input_size), val=True)
    ssd_print(key=mlperf_log.INPUT_SIZE, value=input_size)

    val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.data, "val2017")
    train_annotate = os.path.join(args.data, "annotations/instances_train2017.json")
    train_coco_root = os.path.join(args.data, "train2017")

    cocoGt = COCO(annotation_file=val_annotate)
    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    train_coco = COCODetection(train_coco_root, train_annotate, train_trans)

    #print("Number of labels: {}".format(train_coco.labelnum))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_coco)
    else:
        train_sampler = None
    train_dataloader = DataLoader(train_coco,
                                  batch_size=args.batch_size,
                                  shuffle=(train_sampler is None),
                                  sampler=train_sampler,
                                  num_workers=4)
    # set shuffle=True in DataLoader
    ssd_print(key=mlperf_log.INPUT_SHARD, value=None)
    ssd_print(key=mlperf_log.INPUT_ORDER)
    ssd_print(key=mlperf_log.INPUT_BATCH_SIZE, value=args.batch_size)


    ssd300 = SSD300(train_coco.labelnum)
    if args.checkpoint is not None:
        print("loading model checkpoint", args.checkpoint)
        od = torch.load(args.checkpoint)
        ssd300.load_state_dict(od["model"])
    # if args.param_file is not None:
    #     od = torch.load(args.param_file)
    #     ssd300.load_state_dict(od)

    ssd300.train()
    if use_cuda:
        ssd300.cuda()
    loss_func = Loss(dboxes)
    if use_cuda:
        loss_func.cuda()
    if args.distributed:
        N_gpu = torch.distributed.get_world_size()
    else:
        N_gpu = 1

    # parallelize
    if args.distributed:
        ssd300 = DDP(ssd300)

    global_batch_size = N_gpu * args.batch_size
    current_lr = args.lr * (global_batch_size / 32)
    current_momentum = 0.9
    current_weight_decay = 5e-4
    optim = torch.optim.SGD(ssd300.parameters(), lr=current_lr,
                            momentum=current_momentum,
                            weight_decay=current_weight_decay)
    ssd_print(key=mlperf_log.OPT_NAME, value="SGD")
    ssd_print(key=mlperf_log.OPT_LR, value=current_lr)
    ssd_print(key=mlperf_log.OPT_MOMENTUM, value=current_momentum)
    ssd_print(key=mlperf_log.OPT_WEIGHT_DECAY,
                         value=current_weight_decay)
    eval_points = args.evaluation
    print("epoch", "nbatch", "loss")

    iter_num = args.iteration
    avg_loss = 0.0
    inv_map = {v:k for k,v in val_coco.label_map.items()}
    success = torch.zeros(1)
    if use_cuda:
        success = success.cuda()


    if args.warmup:
        nonempty_imgs = len(train_coco)
        wb = int(args.warmup * nonempty_imgs / (N_gpu*args.batch_size))
        warmup_step = lambda iter_num, current_lr: lr_warmup(optim, wb, iter_num, current_lr, args)
    else:
        warmup_step = lambda iter_num, current_lr: None

    for epoch in range(args.epochs):
        ssd_print(key=mlperf_log.TRAIN_EPOCH, value=epoch)
        # set the epoch for the sampler
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if epoch in args.lr_decay_schedule:
            current_lr *= 0.1
            print("")
            print("lr decay step #{num}".format(num=args.lr_decay_schedule.index(epoch) + 1))
            for param_group in optim.param_groups:
                param_group['lr'] = current_lr
            ssd_print(key=mlperf_log.OPT_LR,
                                 value=current_lr)

        start = time.time()
        for nbatch, (img, img_size, bbox, label) in enumerate(train_dataloader):
            if use_cuda:
                img = img.cuda()
            elif args.use_mkldnn:
                img = img.to_mkldnn()
            img = Variable(img, requires_grad=True)
            ploc, plabel = ssd300(img)
            trans_bbox = bbox.transpose(1,2).contiguous()
            if use_cuda:
                trans_bbox = trans_bbox.cuda()
                label = label.cuda()
            gloc, glabel = Variable(trans_bbox, requires_grad=False), \
                           Variable(label, requires_grad=False)
            loss = loss_func(ploc, plabel, gloc, glabel)

            optim.zero_grad()
            loss.backward()
            warmup_step(iter_num, current_lr)
            optim.step()

            if (iter_num - args.iteration) >= args.perf_prerun_warmup:
                batch_time.update(time.time() - start)

            if not np.isinf(loss.item()): avg_loss = 0.999*avg_loss + 0.001*loss.item()

            print("Iteration: {:6d}, Loss function: {:5.3f}, Average Loss: {:.3f}"\
                        .format(iter_num, loss.item(), avg_loss), end="\r")

            iter_num += 1
            if args.totle_iteration > 0 and (iter_num - args.iteration) >= args.totle_iteration:
                break

            if (iter_num - args.iteration) >= args.perf_prerun_warmup:
                start = time.time()

        if epoch + 1 in eval_points:
            rank = dist.get_rank() if args.distributed else args.local_rank
            if args.distributed:
                world_size = float(dist.get_world_size())
                for bn_name, bn_buf in ssd300.module.named_buffers(recurse=True):
                    if ('running_mean' in bn_name) or ('running_var' in bn_name):
                        dist.all_reduce(bn_buf, op=dist.ReduceOp.SUM)
                        bn_buf /= world_size
            if rank == 0:
                if not args.no_save:
                    print("")
                    print("saving model...")
                    torch.save({"model" : ssd300.state_dict(), "label_map": train_coco.label_info},
                               "./models/iter_{}.pt".format(iter_num))

                if coco_eval(ssd300, val_coco, cocoGt, encoder, inv_map,
                            args.threshold, epoch + 1,iter_num):
                    success = torch.ones(1)
                    if use_cuda:
                        success = success.cuda()
            if args.distributed:
                dist.broadcast(success, 0)
            if success[0]:
                    return True

        batch_size = train_dataloader.batch_size
        latency = batch_time.sum / (iter_num - args.iteration) / batch_size * 1000
        perf = (iter_num - args.iteration) * batch_size/batch_time.sum
        print('training latency %3.0f ms'%latency)
        print('training performance %3.0f fps'%perf)

    return False

def main():
    args = parse_args()

    if args.use_mkldnn and not args.no_cuda:
        logger.error("Mkldnn and CUDA are mutually exclusive")
        return

    if args.local_rank == 0:
        if not os.path.isdir('./models'):
            os.mkdir('./models')

    torch.backends.cudnn.benchmark = True

    if args.totle_iteration > 0:
        assert args.totle_iteration > args.perf_prerun_warmup

    # start timing here
    ssd_print(key=mlperf_log.RUN_START)

    if args.mode == "training":
        success = train300_mlperf_coco(args)
    else:
        success = eval300_mlperf_coco(args)

    # end timing here
    ssd_print(key=mlperf_log.RUN_STOP, value={"success": success})
    ssd_print(key=mlperf_log.RUN_FINAL)

if __name__ == "__main__":
    main()
