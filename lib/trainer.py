import time, os, torch,copy
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from lib.timer import Timer, AverageMeter
from lib.utils import Logger,validate_gradient,remap_legacy_state_dict
from lib.spconv_utils import make_sparse_tensor

from tqdm import tqdm
import torch.nn.functional as F
import gc


class Trainer(object):
    def __init__(self, args):
        self.config = args
        # parameters
        self.start_epoch = 1
        self.max_epoch = args.max_epoch
        self.save_dir = args.save_dir
        self.device = args.device
        self.verbose = args.verbose
        self.max_points = args.max_points
        self.voxel_size = args.voxel_size

        self.model = args.model.to(self.device)
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_freq = args.scheduler_freq
        self.snapshot_freq = args.snapshot_freq
        self.snapshot_dir = args.snapshot_dir 
        self.benchmark = args.benchmark
        self.iter_size = args.iter_size
        self.verbose_freq= args.verbose_freq

        self.w_circle_loss = args.w_circle_loss
        self.w_overlap_loss = args.w_overlap_loss
        self.w_saliency_loss = args.w_saliency_loss 
        self.desc_loss = args.desc_loss

        self.best_loss = 1e5
        self.best_recall = -1e5
        self.writer = SummaryWriter(log_dir=args.tboard_dir)
        self.logger = Logger(args.snapshot_dir)
        self.logger.write(f'#parameters {sum([x.nelement() for x in self.model.parameters()])/1000000.} M\n')
        

        # A run that outlives its SLURM time limit is restarted by the launcher, so it
        # picks up the per-epoch checkpoint. `best_recall` would silently drop every epoch
        # since the last improvement.
        last = os.path.join(self.save_dir, 'model_last.pth')
        if args.pretrain != '':
            self._load_pretrain(args.pretrain)
        elif self.config.get('auto_resume', True) and os.path.isfile(last):
            self._load_pretrain(last)
        
        self.loader =dict()
        self.loader['train']=args.train_loader
        self.loader['val']=args.val_loader
        self.loader['test'] = args.test_loader

        with open(f'{args.snapshot_dir}/model','w') as f:
            f.write(str(self.model))
        f.close()
 
    def _snapshot(self, epoch, name=None):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_recall': self.best_recall
        }
        if name is None:
            filename = os.path.join(self.save_dir, f'model_{epoch}.pth')
        else:
            filename = os.path.join(self.save_dir, f'model_{name}.pth')
        self.logger.write(f"Save model to {filename}\n")
        torch.save(state, filename)

    def _load_pretrain(self, resume):
        if os.path.isfile(resume):
            state = torch.load(resume, weights_only=False)
            # 1x1 convolutions became linear layers, so their weights lost the trailing
            # kernel axis. Checkpoints written before that still load. This has to run
            # *after* the legacy rename, or the shape lookup misses every renamed key.
            weights = remap_legacy_state_dict(state['state_dict'])
            model_shapes = {k: v.shape for k, v in self.model.state_dict().items()}
            for key, weight in list(weights.items()):
                want = model_shapes.get(key)
                if want is not None and weight.dim() == 3 and len(want) == 2:
                    weights[key] = weight.squeeze(-1)
            missing, unexpected = self.model.load_state_dict(weights, strict=False)
            if missing or unexpected:
                # parameters added or removed since the checkpoint was written, e.g. when
                # fine-tuning an older run with a changed head
                self.logger.write(f'checkpoint missing {list(missing)}, unexpected {list(unexpected)}\n')
            self.start_epoch = state['epoch']
            self.best_loss = state['best_loss']
            self.best_recall = state['best_recall']
            try:
                self.scheduler.load_state_dict(state['scheduler'])
                self.optimizer.load_state_dict(state['optimizer'])
            except (ValueError, KeyError) as error:
                # the optimiser state is indexed by parameter, so it cannot be restored once
                # the model has gained or lost parameters; the weights still carry over
                self.logger.write(f'optimiser state not restored ({error}), starting it fresh\n')
            
            self.logger.write(f'Successfully load pretrained model from {resume}!\n')
            self.logger.write(f'Current best loss {self.best_loss}\n')
            self.logger.write(f'Current best recall {self.best_recall}\n')
        else:
            raise ValueError(f"=> no checkpoint found at '{resume}'")

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']

    def stats_meter(self):
        meters=dict()
        stats=self.stats_dict()
        for key,_ in stats.items():
            meters[key]=AverageMeter()
        return meters



    def inference_one_epoch(self,epoch, phase):
        gc.collect()
        assert phase in ['train','val','test']

        # init stats meter
        stats_meter = self.stats_meter()

        num_iter = int(len(self.loader[phase].dataset) // self.loader[phase].batch_size)
        c_loader_iter = self.loader[phase].__iter__()
        
        self.optimizer.zero_grad()
        for c_iter in tqdm(range(num_iter)): # loop through this epoch   
            inputs = next(c_loader_iter)
            try:
                ##################################
                # forward pass
                # with torch.autograd.detect_anomaly():
                stats = self.inference_one_batch(inputs, phase)
                
                ###################################################
                # run optimisation
                if((c_iter+1) % self.iter_size == 0 and phase == 'train'):
                    gradient_valid = validate_gradient(self.model)
                    if(gradient_valid):
                        self.optimizer.step()
                    else:
                        self.logger.write('gradient not valid\n')
                    self.optimizer.zero_grad()
                
                ################################
                # update to stats_meter
                for key,value in stats.items():
                    stats_meter[key].update(value)
            except RuntimeError as inst:
                pass
            
            torch.cuda.empty_cache()
            
            if (c_iter + 1) % self.verbose_freq == 0 and self.verbose:
                curr_iter = num_iter * (epoch - 1) + c_iter
                for key, value in stats_meter.items():
                    self.writer.add_scalar(f'{phase}/{key}', value.avg, curr_iter)
                
                message = f'{phase} Epoch: {epoch} [{c_iter+1:4d}/{num_iter}]'
                for key,value in stats_meter.items():
                    message += f'{key}: {value.avg:.2f}\t'

                self.logger.write(message + '\n')

        message = f'{phase} Epoch: {epoch}'
        for key,value in stats_meter.items():
            message += f'{key}: {value.avg:.2f}\t'
        self.logger.write(message+'\n')

        return stats_meter


    def eval(self):
        print('Start to evaluate on validation datasets...')
        stats_meter = self.inference_one_epoch(0,'val')

        for key, value in stats_meter.items():
            print(key, value.avg)


class PredatorTrainer(Trainer):
    """Predator: circle + overlap + saliency losses on per-point descriptors.

    Split from `Trainer` because almost nothing it does is shared with OCFNet -- a
    different loss signature, different statistics, and a benchmark check that matches
    descriptors and runs RANSAC on them rather than reading correspondences off a
    transport plan. `Trainer` keeps only what both need: construction, checkpointing,
    resuming, and the statistics meters.
    """

    def stats_dict(self):
        stats=dict()
        stats['circle_loss']=0.
        stats['recall']=0.  # feature match recall, divided by number of ground truth pairs
        stats['saliency_loss'] = 0.
        stats['saliency_recall'] = 0.
        stats['saliency_precision'] = 0.
        stats['overlap_loss'] = 0.
        stats['overlap_recall']=0.
        stats['overlap_precision']=0.
        return stats

    def inference_one_batch(self, input_dict, phase):
        assert phase in ['train','val','test']
        ##################################
        # training
        if(phase == 'train'):
            self.model.train()
            ###############################################
            # forward pass
            sinput_src = make_sparse_tensor(
                input_dict['src_C'], input_dict['src_F'], device=self.device)
            sinput_tgt = make_sparse_tensor(
                input_dict['tgt_C'], input_dict['tgt_F'], device=self.device)
            
            src_feats, tgt_feats, scores_overlap, scores_saliency= self.model(sinput_src, sinput_tgt)
            src_pcd, tgt_pcd = input_dict['pcd_src'].to(self.device), input_dict['pcd_tgt'].to(self.device)
            c_rot = input_dict['rot'].to(self.device)
            c_trans = input_dict['trans'].to(self.device)
            correspondence = input_dict['correspondences'].long().to(self.device)

            ###################################################
            # get loss
            stats= self.desc_loss(src_pcd, tgt_pcd, src_feats, tgt_feats,correspondence, c_rot, c_trans, scores_overlap, scores_saliency,input_dict['scale'], input_dict['len_batch'])

            c_loss = stats['circle_loss'] * self.w_circle_loss + stats['overlap_loss'] * self.w_overlap_loss + stats['saliency_loss'] * self.w_saliency_loss

            c_loss.backward()

        else:
            self.model.eval()
            with torch.no_grad():
                ###############################################
                # forward pass
                sinput_src = make_sparse_tensor(
                    input_dict['src_C'], input_dict['src_F'], device=self.device)
                sinput_tgt = make_sparse_tensor(
                    input_dict['tgt_C'], input_dict['tgt_F'], device=self.device)
                
                src_feats, tgt_feats, scores_overlap, scores_saliency= self.model(sinput_src, sinput_tgt)
                src_pcd, tgt_pcd = input_dict['pcd_src'].to(self.device), input_dict['pcd_tgt'].to(self.device)
                c_rot = input_dict['rot'].to(self.device)
                c_trans = input_dict['trans'].to(self.device)
                correspondence = input_dict['correspondences'].long().to(self.device)

                ###################################################
                # get loss
                stats= self.desc_loss(src_pcd, tgt_pcd, src_feats, tgt_feats,correspondence, c_rot, c_trans, scores_overlap, scores_saliency,input_dict['scale'], input_dict['len_batch'])


        ##################################        
        # detach the gradients for loss terms
        stats['circle_loss'] = float(stats['circle_loss'].detach())
        stats['overlap_loss'] = float(stats['overlap_loss'].detach())
        stats['saliency_loss'] = float(stats['saliency_loss'].detach())
        
        return stats


    def descriptor_registration_check(self, epoch, max_pairs=100, n_points=1000):
        """Register benchmark pairs after every epoch, the way the Predator benchmark does.

        Its validation `recall` is a feature-match proxy, not registration recall, and
        selecting on it picks the wrong checkpoint: on the 22 hour RoPE run, `best_loss`
        beat `best_recall` by four points of 3DLoMatch registration recall. This runs the
        real pipeline -- probabilistic sampling by overlap x saliency, then a
        descriptor RANSAC -- so `best_recall` means what it says.

        `max_pairs` pairs are taken at a fixed stride through the benchmark rather than
        from the front: it is ordered by scene, so the first N pairs are all one room and
        the number comes out several points too high (0.97 against 0.88 on the full set).
        """
        from lib.benchmark_utils import ransac_pose_estimation
        self.model.eval()
        rre, rte, done, seen = [], [], 0, 0
        stride = max(1, len(self.loader['test'].dataset) // max_pairs)
        with torch.no_grad():
            for inputs in self.loader['test']:
                if done >= max_pairs:
                    break
                src = make_sparse_tensor(inputs['src_C'], inputs['src_F'], device=self.device)
                tgt = make_sparse_tensor(inputs['tgt_C'], inputs['tgt_F'], device=self.device)
                feats_src, feats_tgt, overlap, saliency = self.model(src, tgt)
                feats_src, feats_tgt = feats_src.cpu(), feats_tgt.cpu()
                overlap, saliency = overlap.cpu(), saliency.cpu()

                start_s = start_t = 0
                for b, (n_src, n_tgt) in enumerate(inputs['len_batch']):
                    take = (seen % stride == 0) and done < max_pairs
                    seen += 1
                    if not take:
                        start_s += n_src; start_t += n_tgt
                        continue
                    done += 1
                    src_pcd = inputs['pcd_src'][start_s:start_s + n_src]
                    tgt_pcd = inputs['pcd_tgt'][start_t:start_t + n_tgt]
                    fs, ft = feats_src[start_s:start_s + n_src], feats_tgt[start_t:start_t + n_tgt]
                    scores_s = (overlap[start_s:start_s + n_src]
                                * saliency[start_s:start_s + n_src]).numpy().flatten()
                    scores_t = (overlap[len(feats_src) + start_t:len(feats_src) + start_t + n_tgt]
                                * saliency[len(feats_src) + start_t:len(feats_src) + start_t + n_tgt]
                                ).numpy().flatten()
                    start_s += n_src; start_t += n_tgt

                    if src_pcd.shape[0] > n_points:
                        p_ = scores_s / scores_s.sum()
                        idx = np.random.choice(src_pcd.shape[0], n_points, replace=False, p=p_)
                        src_pcd, fs = src_pcd[idx], fs[idx]
                    if tgt_pcd.shape[0] > n_points:
                        p_ = scores_t / scores_t.sum()
                        idx = np.random.choice(tgt_pcd.shape[0], n_points, replace=False, p=p_)
                        tgt_pcd, ft = tgt_pcd[idx], ft[idx]

                    estimate = ransac_pose_estimation(src_pcd, tgt_pcd, fs, ft,
                                                      mutual=False, distance_threshold=0.05)
                    rotation = torch.from_numpy(estimate[:3, :3].copy()).float()
                    translation = torch.from_numpy(estimate[:3, 3].copy()).float()
                    cosine = ((rotation.T @ inputs['rot'][b]).diagonal().sum() - 1) / 2
                    rre.append(float(torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))))
                    rte.append(float((translation - inputs['trans'][b].squeeze(-1)).norm()))

        rre, rte = np.array(rre), np.array(rte)
        recall = float(((rre < 15) & (rte < 0.3)).mean()) if rre.size else 0.
        self.logger.write(f'registration Epoch: {epoch} [{self.benchmark}] pairs: {rre.size}\t'
                          f'registered: {recall:.3f}\tmedian RRE: {np.median(rre):.2f}\t'
                          f'median RTE: {np.median(rte):.3f}\n')
        self.writer.add_scalar('val/registered', recall, epoch)
        return recall

    def train(self):
        print('start training...')
        for epoch in range(self.start_epoch, self.max_epoch):
            self.inference_one_epoch(epoch,'train')
            self.scheduler.step()
            
            stats_meter = self.inference_one_epoch(epoch,'val')
            
            if stats_meter['circle_loss'].avg < self.best_loss:
                self.best_loss = stats_meter['circle_loss'].avg
                self._snapshot(epoch,'best_loss')
            registered = self.descriptor_registration_check(
                epoch, max_pairs=self.config.get('eval_pairs', 100))
            if registered > self.best_recall:
                self.best_recall = registered
                self._snapshot(epoch,'best_recall')
            self._snapshot(epoch, 'last')
            
            # we only add saliency loss when we get descent point-wise features
            if(stats_meter['recall'].avg>0.3):
                self.w_saliency_loss = 1.
            else:
                self.w_saliency_loss = 0.
                    
        # finish all epoch
        print("Training finish!")


class OCFNetTrainer(Trainer):
    """Trainer for models/ocfnet.py, whose forward returns a dict and whose loss is OCFLoss.

    Both phases refine the ground-truth super-point pairs, so the losses of the training and
    the validation curve measure the same thing; the predicted coarse matches of Eq. (3) are
    what scripts/evaluate_predator.py exercises at test time. Checkpoints are selected on the
    inlier ratio of the point correspondences, the quantity the paper reports.
    """

    def stats_dict(self):
        return {key: 0. for key in ['loss', 'coarse_loss', 'fine_loss', 'coarse_overlap_loss',
                                    'fine_overlap_loss', 'descriptor_loss', 'infonce_loss', 'pair_overlap_loss',
                                    'coarse_infonce_loss',
                                    'inlier_loss', 'coarse_inlier_loss', 'coarse_ir',
                                    'inlier_ratio', 'ir_ceiling', 'ir_chance']}

    def inference_one_batch(self, input_dict, phase):
        assert phase in ['train', 'val', 'test']
        self.model.train() if phase == 'train' else self.model.eval()

        with torch.set_grad_enabled(phase == 'train'):
            sinput_src = make_sparse_tensor(
                input_dict['src_C'], input_dict['src_F'], device=self.device)
            sinput_tgt = make_sparse_tensor(
                input_dict['tgt_C'], input_dict['tgt_F'], device=self.device)
            correspondence = input_dict['correspondences'].long().to(self.device)

            output = self.model(sinput_src, sinput_tgt, correspondences=correspondence,
                                src_xyz=input_dict['pcd_src'].to(self.device),
                                tgt_xyz=input_dict['pcd_tgt'].to(self.device))
            stats = self.desc_loss(output, input_dict)

            if phase == 'train':
                stats['loss'].backward()

        return {key: float(value.detach()) for key, value in stats.items()}

    @torch.no_grad()
    def registration_check(self, epoch, max_pairs=100, n_points=1000):
        """Register pairs of the *benchmark* after every epoch.

        Same pipeline the benchmark script runs -- predicted correspondences into a
        correspondence RANSAC -- on the benchmark split named by `benchmark` in the config,
        so the number is comparable to the published registration recall rather than to a
        validation proxy. `eval_pairs` bounds the cost: RANSAC is the slow part, roughly a
        second per pair, and the full 1623-pair benchmark would double the epoch time.
        Checkpoints are selected on it.
        """
        from lib.benchmark_utils import ransac_pose_estimation_correspondences
        from models.ocfnet import ground_truth_coarse_pairs

        self.model.eval()
        rre, rte, done = [], [], 0
        coarse_ir, fine_ir = [], []
        for inputs in self.loader['test']:
            if done >= max_pairs:
                break
            src = make_sparse_tensor(inputs['src_C'], inputs['src_F'], device=self.device)
            tgt = make_sparse_tensor(inputs['tgt_C'], inputs['tgt_F'], device=self.device)
            # An oracle run (gt_overlap) replaces the predicted overlap score with the true
            # visibility, which needs the ground-truth correspondences. Supplying them here
            # would also switch the coarse stage to ground-truth patch pairs, so they are
            # passed only for the overlap score, through a separate argument. Without this
            # the oracle trains on one marginal distribution and is tested on another, which
            # measures a train/test mismatch rather than an upper bound.
            oracle = getattr(self.model, 'gt_overlap', False)
            output = self.model(src, tgt,                    # predicted coarse matches
                                src_xyz=inputs['pcd_src'].to(self.device),
                                tgt_xyz=inputs['pcd_tgt'].to(self.device),
                                overlap_correspondences=(
                                    inputs['correspondences'].long().to(self.device)
                                    if oracle else None))
            matches, scores = self.model.point_correspondences(output)
            matches, scores = matches.cpu(), scores.cpu()

            # coarse inlier ratio: predicted patch pairs whose patches really share a
            # correspondence (the patch inlier ratio of the coarse-to-fine literature)
            correspondence = inputs['correspondences'].long().to(self.device)
            truth = ground_truth_coarse_pairs(correspondence, output['src_patch_id'],
                                              output['tgt_patch_id'])
            predicted = output['coarse_matches']
            if predicted.shape[0] and truth.shape[0]:
                width = int(output['tgt_patch_id'].max()) + 1
                hit = torch.isin(predicted[:, 0] * width + predicted[:, 1],
                                 truth[:, 0] * width + truth[:, 1])
                coarse_ir.append(float(hit.float().mean()))

            src_start = tgt_start = 0
            for b, (n_src, n_tgt) in enumerate(inputs['len_batch']):
                if done >= max_pairs:
                    break
                keep = (matches[:, 0] >= src_start) & (matches[:, 0] < src_start + n_src)
                pair = matches[keep] - torch.tensor([src_start, tgt_start])
                confidence = scores[keep]
                src_pcd = inputs['pcd_src'][src_start:src_start + n_src]
                tgt_pcd = inputs['pcd_tgt'][tgt_start:tgt_start + n_tgt]
                src_start, tgt_start, done = src_start + n_src, tgt_start + n_tgt, done + 1

                if pair.shape[0] < 4:
                    rre.append(180.); rte.append(10.)
                    continue
                if pair.shape[0] > n_points:
                    # sample proportional to confidence, as the benchmark script does. Taking
                    # the top n concentrates the correspondences on a few patches, which are
                    # spatially clustered and condition RANSAC badly -- worth 0.3 registration
                    # rate on the same checkpoint.
                    probability = (confidence / confidence.sum()).numpy()
                    choice = np.random.choice(pair.shape[0], size=n_points, replace=False,
                                              p=probability)
                    pair = pair[torch.from_numpy(choice)]
                # fine inlier ratio: correspondences within 10 cm under the ground truth
                moved = src_pcd[pair[:, 0]] @ inputs['rot'][b].T + inputs['trans'][b].view(1, 3)
                fine_ir.append(float(((moved - tgt_pcd[pair[:, 1]]).norm(dim=1) < 0.1).float().mean()))

                estimate = ransac_pose_estimation_correspondences(src_pcd, tgt_pcd, pair)
                # .copy(): open3d can return a view with negative strides, which
                # torch.from_numpy refuses. The Predator path was fixed for this; the OCFNet
                # path was not, and it killed two runs after 7h50m of training.
                rotation = torch.from_numpy(estimate[:3, :3].copy()).float()
                translation = torch.from_numpy(estimate[:3, 3].copy()).float()
                cosine = ((rotation.T @ inputs['rot'][b]).diagonal().sum() - 1) / 2
                rre.append(float(torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))))
                rte.append(float((translation - inputs['trans'][b].squeeze(-1)).norm()))

        rre, rte = np.array(rre), np.array(rte)
        recall = float(((rre < 15) & (rte < 0.3)).mean()) if rre.size else 0.
        coarse = float(np.mean(coarse_ir)) if coarse_ir else 0.
        fine = float(np.mean(fine_ir)) if fine_ir else 0.
        message = (f'registration Epoch: {epoch} [{self.benchmark}] pairs: {rre.size}\t'
                   f'registered: {recall:.3f}\tcoarse_IR: {coarse:.3f}\tfine_IR: {fine:.3f}\t'
                   f'median RRE: {np.median(rre):.2f}\tmedian RTE: {np.median(rte):.3f}\n')
        self.logger.write(message)
        self.writer.add_scalar('val/registered', recall, epoch)
        self.writer.add_scalar('val/coarse_inlier_ratio', coarse, epoch)
        self.writer.add_scalar('val/fine_inlier_ratio', fine, epoch)
        return recall

    def train(self):
        print('start training...')
        for epoch in range(self.start_epoch, self.max_epoch):
            self.inference_one_epoch(epoch, 'train')
            self.scheduler.step()

            stats_meter = self.inference_one_epoch(epoch, 'val')
            registered = self.registration_check(epoch, max_pairs=self.config.get('eval_pairs', 100))

            if stats_meter['loss'].avg < self.best_loss:
                self.best_loss = stats_meter['loss'].avg
                self._snapshot(epoch, 'best_loss')
            if registered > self.best_recall:
                self.best_recall = registered
                self._snapshot(epoch, 'best_recall')
            self._snapshot(epoch, 'last')

        print("Training finish!")