import errno
import sys
import os.path as osp
import torch.utils.data as data
import os
import torch
import numpy as np
import random
from BERT_token_process_ICFG import CUHKPEDES_BERT_token
from torch.utils.data.distributed import DistributedSampler

def data_config(dir, batch_size, split, max_length, embedding_type,transform):
    print("The word length is", max_length)
    if embedding_type == 'BERT':
        print("The word embedding type is BERT")
        data_split = CUHKPEDES_BERT_token(split, max_length, transform)
    print("the number of", split, ":", len(data_split))
    if split == 'train':
        shuffle = True
        #data_sampler = DistributedSampler(data_split)
        #loader = data.DataLoader(data_split, sampler=data_sampler, batch_size=batch_size, num_workers=8)
        loader = data.DataLoader(data_split, batch_size=batch_size, num_workers=8)
    else:
        #shuffle = False
        #data_sampler = DistributedSampler(data_split)
        loader = data.DataLoader(data_split, batch_size=batch_size, num_workers=8)
    #data_sampler = DistributedSampler(data_split)
    #loader = data.DataLoader(data_split, sampler=data_sampler, batch_size=batch_size, num_workers=8)
    return loader

def optimizer_function(args, model):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.adam_lr, betas=(args.adam_alpha, args.adam_beta), eps=args.epsilon, weight_decay=0.01)
        print("optimizer is：Adam")
    return optimizer

def lr_scheduler(optimizer, args):

    if args.lr_decay_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min', factor=args.lr_decay_ratio,
                                                           patience=5, min_lr=args.end_lr)
        print("lr_scheduler is ReduceLROnPlateau")
    else:
        if '_' in args.epoches_decay:
            epoches_list = args.epoches_decay.split('_')
            epoches_list = [int(e) for e in epoches_list]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, epoches_list, gamma=args.lr_decay_ratio)
            print("lr_scheduler is MultiStepLR")
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(args.epoches_decay), gamma=args.lr_decay_ratio)
            print("lr_scheduler is StepLR")
    return scheduler

def load_checkpoint(model,resume):
    start_epoch=0

    if os.path.isfile(resume):
        checkpoint = torch.load(resume)
        # checkpoint= torch.load(resume, map_location='cuda:0')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print('Load checkpoint at epoch %d.' % (start_epoch))
    return start_epoch,model

class AverageMeter(object):
    """
    Computes and stores the averate and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py #L247-262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += n * val
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, epoch, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    filename = os.path.join(dst, str(epoch)) + '.pth.tar'
    torch.save(state, filename)

def gradual_warmup(epoch,init_lr,optimizer,epochs):
    lr = init_lr
    if epoch < epochs:
        warmup_percent_done = (epoch+1) / epochs
        warmup_learning_rate = init_lr * warmup_percent_done
        lr = warmup_learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def compute_topk(query, gallery, target_query, target_gallery, k=[1,10], reverse=False):
    result = []
    query = query / (query.norm(dim=1,keepdim=True)+1e-12)
    gallery = gallery / (gallery.norm(dim=1,keepdim=True)+1e-12)
    sim_cosine = torch.matmul(query, gallery.t())
    result.extend(topk(sim_cosine, target_gallery, target_query, k))
    if reverse:
        result.extend(topk(sim_cosine, target_query, target_gallery, k, dim=0))
    return result


def topk(sim, target_gallery, target_query, k=[1,10], dim=1):
    result = []
    maxk = max(k)
    size_total = len(target_gallery)
    _, pred_index = sim.topk(maxk, dim, True, True)
    pred_labels = target_gallery[pred_index]
    if dim == 1:
        pred_labels = pred_labels.t()
    correct = pred_labels.eq(target_query.view(1,-1).expand_as(pred_labels))

    for topk in k:
        correct_k = torch.sum(correct[:topk], dim=0)
        correct_k = torch.sum(correct_k > 0).float()
        result.append(correct_k * 100 / size_total)
    return result

def check_exists(root):
    if os.path.exists(root):
        return True
    return False

def load_embedding(path):
    word_embedding=torch.from_numpy(np.load(path))
    (vocab_size,embedding_size)=word_embedding.shape
    print('Load word embedding,the shape of word embedding is [{},{}]'.format(vocab_size,embedding_size))
    return word_embedding

def load_part_model(model,path):
    model_dict = model.state_dict()
    checkpoint = torch.load(path)
    pretrained_dict = checkpoint["state_dict"]
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def test_map(query_feature, query_label, gallery_feature, gallery_label):
    query_feature = query_feature / (query_feature.norm(dim=1, keepdim=True) + 1e-12)
    gallery_feature = gallery_feature / (gallery_feature.norm(dim=1, keepdim=True) + 1e-12)
    #qf1 = qf1 / (qf1.norm(dim=1, keepdim=True) + 1e-12)
    #gf1 = gf1 / (gf1.norm(dim=1, keepdim=True) + 1e-12)
    #qf2 = qf2 / (qf2.norm(dim=1, keepdim=True) + 1e-12)
    #gf2 = gf2 / (gf2.norm(dim=1, keepdim=True) + 1e-12)
    #qf3 = qf3 / (qf3.norm(dim=1, keepdim=True) + 1e-12)
    #gf3 = gf3 / (gf3.norm(dim=1, keepdim=True) + 1e-12)
    #qf4 = qf4 / (qf4.norm(dim=1, keepdim=True) + 1e-12)
    #gf4 = gf4 / (gf4.norm(dim=1, keepdim=True) + 1e-12)
    #qf5 = qf5 / (qf5.norm(dim=1, keepdim=True) + 1e-12)
    #gf5 = gf5 / (gf5.norm(dim=1, keepdim=True) + 1e-12)
    #qf6 = qf6 / (qf6.norm(dim=1, keepdim=True) + 1e-12)
    #gf6 = gf6 / (gf6.norm(dim=1, keepdim=True) + 1e-12)
    #global_query_features = global_q_f / (global_q_f.norm(dim=1, keepdim=True) + 1e-12)
    #global_gallery_features = global_g_f / (global_g_f.norm(dim=1, keepdim=True) + 1e-12)
    #for i in range(len(query_branch)):
    #    query_branch[i] = query_branch[i] / (query_branch[i].norm(dim=1, keepdim=True) + 1e-12)
    #    gallery_branch[i] = gallery_branch[i] / (gallery_branch[i].norm(dim=1, keepdim=True) + 1e-12)
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)

        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    CMC = CMC.float()
    CMC = CMC / len(query_label)
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))
    return CMC[0], CMC[4], CMC[9], ap / len(query_label)

def evaluate(qf, ql, gf, gl):
    query = qf.view(-1, 1)
    #q1 = qf1.view(-1, 1)
    #q2 = qf2.view(-1, 1)
    #q3 = qf3.view(-1, 1)
    #q4 = qf4.view(-1, 1)
    #q5 = qf5.view(-1, 1)
    #q6 = qf6.view(-1, 1)
    #gqf = gqf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    #score1 = torch.mm(gf1, q1)
    #score1 = score1.squeeze(1).cpu()
    #score2 = torch.mm(gf2, q2)
    #score2 = score2.squeeze(1).cpu()
    #score3 = torch.mm(gf3, q3)
    #score3 = score3.squeeze(1).cpu()
    #score4 = torch.mm(gf4, q4)
    #score4 = score4.squeeze(1).cpu()
    #score5 = torch.mm(gf5, q5)
    #score5 = score5.squeeze(1).cpu()
    #score6 = torch.mm(gf6, q6)
    #score6 = score6.squeeze(1).cpu()
    score = score.numpy()
    #score1 = score1.numpy()
    #score2 = score2.numpy()
    #score3 = score3.numpy()
    #score4 = score4.numpy()
    #score5 = score5.numpy()
    #score6 = score6.numpy()
    #score = score + score1 + score2 + score3 + score4 + score5 + score6
    #score2 = torch.mm(ggf, gqf)
    #score2 = score2.squeeze(1).cpu()
    #score2 = score2.numpy()
    #score = score + score2
    index = np.argsort(score)
    index = index[::-1]
    '''
    for i in range(10):
        print(gl[int(index[i])])
    '''
    gl=gl.data.cpu().numpy()
    ql=ql.data.cpu().numpy()
    query_index = np.argwhere(gl == ql)
    CMC_tmp = compute_mAP(index, query_index)
    return CMC_tmp


def compute_mAP(index, good_index):
    #print(index[:10])
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc
    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()
    #print(rows_good)

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
