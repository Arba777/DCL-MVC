import copy
import torch
from torch import optim
from torch.utils.data import DataLoader
from dataloader import build_dataset
import torch.nn as nn
from base_model_att_cont import DVIMC
from evaluate import evaluate
from base_fn import kl_term, vade_trick, coherence_function_att, contrastiveLoss, cluster_contrastive_loss, prototype_contrastive_loss, get_contrastive_weights
import numpy as np
import random
import argparse
from sklearn.cluster import KMeans
import torch.nn.functional as F
import time


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_view_value(R, H, view):
    """
    softmax(-MMD)
    R: list of [N, D] 
    H:   [N, D] 
    returns:  [V] tensor, sum to 1, no NaN/Inf, requires_grad=False
    """
    with torch.no_grad():
        num_views = len(R)
        mmd_list = []

        D = H.size(1)
        global_sim = (H @ H.t()) / D
        for v in range(num_views):
            view_sim    = (R[v] @ R[v].t()) / D
            related_sim = (R[v] @ H.t())   / D

            mmd = torch.mean(view_sim + global_sim - 2 * related_sim)
            mmd_list.append(mmd)
        mmds = torch.stack(mmd_list)             # [V]

        w = F.softmax(-mmds, dim=0)              # [V]

        w = torch.nan_to_num(w,
                             nan=1.0/num_views,    
                             posinf=1.0,           
                             neginf=0.0)

        w = w / (w.sum() + 1e-9)
        return w

def initialization(model, sv_loaders, cmv_data, args):
    print('Initializing......')
    criterion = nn.MSELoss()
    for v in range(args.num_views):
        optimizer = optim.Adam([{"params": model.encoders[f'view_{v}'].parameters(), 'lr': 0.001},
                                {"params": model.decoders[f'view_{v}'].parameters(), 'lr': 0.001},
                                ])
        for e in range(1, args.initial_epochs + 1):
            for batch_idx, xv in enumerate(sv_loaders[v]):
                optimizer.zero_grad()
                batch_size = xv.shape[0]
                xv = xv.reshape(batch_size, -1).to(args.device)
                _, xvr = model.sv_encode(xv, v)
                view_rec_loss = criterion(xvr, xv)
                view_rec_loss.backward()
                optimizer.step()
    with torch.no_grad():
        initial_data = [torch.tensor(csv_data, dtype=torch.float32).to(args.device) for csv_data in cmv_data]
        latent_representation_list = model.mv_encode(initial_data)
        assert len(latent_representation_list) == args.num_views
        fused_latent_representations = sum(latent_representation_list) / len(latent_representation_list)
        fused_latent_representations = fused_latent_representations.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=args.class_num, n_init=10)
        kmeans.fit(fused_latent_representations)
        model.prior_mu.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(args.device)


def train(model, optimizer, scheduler, imv_loader, args):
    print('Training......')

    total_train_start = time.perf_counter()

    epoch_times = []
    batch_times = []
    eval_times = []

    eval_data = copy.deepcopy(imv_loader.dataset.data_list)
    eval_mask = copy.deepcopy(imv_loader.dataset.mask_list)
    for v in range(args.num_views):
        eval_data[v] = torch.tensor(eval_data[v], dtype=torch.float32).to(args.device)
        eval_mask[v] = torch.tensor(eval_mask[v], dtype=torch.float32).to(args.device)
    eval_labels = imv_loader.dataset.labels

    if args.likelihood == 'Bernoulli':
        likelihood_fn = nn.BCEWithLogitsLoss(reduction='none')
    else:
        likelihood_fn = nn.MSELoss(reduction='none')


    best_acc = 0.0
    best_nmi = 0.0
    best_ari = 0.0
    best_pur = 0.0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        epoch_loss = []
        recEpoch_loss = []
        coherenceEpoch_loss = []
        contraEpoch_loss = []
        cluster_contra_loss_list = [] 
        proto_contra_loss_list = [] 

        for batch_idx, (batch_data, batch_mask) in enumerate(imv_loader):
            batch_start = time.perf_counter()

            optimizer.zero_grad()
            batch_data = [sv_d.to(args.device) for sv_d in batch_data]
            batch_mask = [sv_m.to(args.device) for sv_m in batch_mask]
            vs_mus, vs_vars, aggregated_mu, aggregated_var, xr_list, vade_z_sample, att_weight, R, H = model(batch_data, batch_mask) 

            qc_x = vade_trick(vade_z_sample, model.prior_weight, model.prior_mu, model.prior_var)
            z_loss, c_loss = kl_term(aggregated_mu, aggregated_var, qc_x, model.prior_weight, model.prior_mu, model.prior_var)
            kl_loss = z_loss + c_loss

            rec_term = []
            for v in range(args.num_views):
                sv_rec = torch.sum(likelihood_fn(xr_list[v], batch_data[v]), dim=1)  # ( Batch size * Dv )
                exist_rec = sv_rec * batch_mask[v].squeeze()
                view_rec_loss = torch.mean(exist_rec)
                rec_term.append(view_rec_loss)
            rec_loss = sum(rec_term)

            # coherence_loss = coherence_function(vs_mus, vs_vars, aggregated_mu, aggregated_var, batch_mask)
            coherence_loss = coherence_function_att(vs_mus, vs_vars, aggregated_mu, aggregated_var, att_weight, batch_mask)

           
            with torch.no_grad():
                w = compute_view_value(R, H, args.num_views)
                # print("w:",w)
            contrastive_term = []
            for v in range(args.num_views):
                mask_v = batch_mask[v].squeeze()
                
                contrastive_temp = contrastiveLoss(H, R[v], mask_v, args.temperature, w[v]) # args.temperature还没定义
                contrastive_term.append(contrastive_temp)
            contrastive_loss = sum(contrastive_term)

            
            cluster_contrast_loss = cluster_contrastive_loss(model.prior_mu, temperature=args.temperature_cluster)

            
            prototype_contrast_loss = prototype_contrastive_loss(
                z=vade_z_sample, 
                qc_x=qc_x, 
                mog_mu=model.prior_mu, 
                temperature=args.temperature_prototype, 
                confidence_threshold=args.confidence_threshold, 
                epoch=epoch, 
                total_epochs=args.epochs
            )

            
            contrastive_weight = get_contrastive_weights(epoch, args.epochs)

            
            batch_loss = (rec_loss + kl_loss + args.alpha * coherence_loss + args.beta * contrastive_loss + 
                          contrastive_weight * cluster_contrast_loss +
                          contrastive_weight * prototype_contrast_loss)
            # batch_loss = rec_loss + kl_loss 
            # batch_loss = rec_loss + kl_loss + contrastive_loss
            epoch_loss.append(batch_loss.item())
            recEpoch_loss.append(rec_loss.item())
            coherenceEpoch_loss.append(coherence_loss.item())
            contraEpoch_loss.append(contrastive_loss.item())
            cluster_contra_loss_list.append(cluster_contrast_loss.item()) 
            proto_contra_loss_list.append(prototype_contrast_loss.item()) 


            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            model.prior_weight.data = model.prior_weight.data / model.prior_weight.data.sum()

            batch_end = time.perf_counter()
            batch_times.append(batch_end - batch_start)

        scheduler.step()
        overall_loss = sum(epoch_loss) / len(epoch_loss)
        
        overallRec_loss = sum(recEpoch_loss) / len(recEpoch_loss)
        overallCoh_loss = sum(coherenceEpoch_loss) / len(coherenceEpoch_loss)
        overallContr_loss = sum(contraEpoch_loss) / len(contraEpoch_loss)
        overallClusterContr_loss = sum(cluster_contra_loss_list) / len(cluster_contra_loss_list)  
        overallProtoContr_loss = sum(proto_contra_loss_list) / len(proto_contra_loss_list) 

        eval_start = time.perf_counter()
        
        with torch.no_grad():
            _, _, aggregated_mu, _, _, _, _, _, _ = model(eval_data, eval_mask)

            mog_weight = model.prior_weight.data.detach().cpu()
            mog_mu = model.prior_mu.data.detach().cpu()
            mog_var = model.prior_var.data.detach().cpu()
            aggregated_mu = aggregated_mu.detach().cpu()

            c_assignment = vade_trick(aggregated_mu, mog_weight, mog_mu, mog_var)
            predict = torch.argmax(c_assignment, dim=1).numpy()
            acc, nmi, ari, pur = evaluate(eval_labels, predict)

            
            if acc > best_acc:
                best_acc = acc
            if nmi > best_nmi:
                best_nmi = nmi
            if ari > best_ari:
                best_ari = ari
            if pur > best_pur:
                best_pur = pur
        eval_end = time.perf_counter()
        current_eval_time = eval_end - eval_start
        eval_times.append(current_eval_time)

        epoch_end = time.perf_counter()
        current_epoch_time = epoch_end - epoch_start
        epoch_times.append(current_epoch_time)

        if epoch % args.interval == 0 or epoch == args.epochs:
            print(
                f'Epoch {epoch:>3}/{args.epochs}  '
                f'Loss:{overall_loss:.2f}  '
                f'ACC:{acc * 100:.2f}  NMI:{nmi * 100:.2f}  '
                f'ARI:{ari * 100:.2f}  PUR:{pur * 100:.2f}  '
                f'EpochTime:{current_epoch_time:.3f}s  '
                f'EvalTime:{current_eval_time:.3f}s'
            )


    print('Finish training')
    total_train_end = time.perf_counter()
    total_train_time = total_train_end - total_train_start

    timing_stats = {
        "total_train_time": total_train_time,
        "avg_epoch_time": sum(epoch_times) / len(epoch_times),
        "avg_batch_time": sum(batch_times) / len(batch_times),
        "total_eval_time": sum(eval_times),
        "avg_eval_time": sum(eval_times) / len(eval_times),
    }

    return (best_acc, best_nmi, best_ari, best_pur), (acc, nmi, ari, pur), timing_stats


def main(args):
    torch.autograd.set_detect_anomaly(True)
    


    test_record = {"ACC": [], "NMI": [], "ARI": [], "PUR": []}
    best_record = {"ACC": [], "NMI": [], "ARI": [], "PUR": []}
    time_record = {
                        "total_train_time": [],
                        "avg_epoch_time": [],
                        "avg_batch_time": [],
                        "total_eval_time": [],
                        "avg_eval_time": []
                    }
    
    for t in range(1, args.test_times + 1):
        print(f'Test {t}')
        np.random.seed(t)
        random.seed(t)
        complete_data, imv_dataset, sv_datasets = build_dataset(args)
        # cmv_data = complete_data    #datasets
        cmv_data = complete_data['data']  #dataloader
        setup_seed(args.seed)
        imv_loader = DataLoader(imv_dataset, batch_size=args.batch_size, shuffle=True)
        sv_loaders = [DataLoader(sv_dataset, batch_size=args.batch_size, shuffle=True) for sv_dataset in sv_datasets]
        model = DVIMC(args).to(args.device)

        optimizer = optim.Adam(
            [{"params": model.encoders.parameters(), 'lr': args.learning_rate},
             {"params": model.decoders.parameters(), 'lr': args.learning_rate},
             {"params": model.prior_weight, 'lr': args.prior_learning_rate},
             {"params": model.prior_mu, 'lr': args.prior_learning_rate},
             {"params": model.prior_var, 'lr': args.prior_learning_rate},
             {"params": model.attention.parameters(), 'lr': args.learning_rate},
             {"params": model.common_information_module.parameters(), 'lr': args.learning_rate},
             {"params": model.cross_mu.parameters(), 'lr': args.learning_rate},
             {"params": model.cross_var.parameters(), 'lr': args.learning_rate}
             ])
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_factor)
        initialization(model, sv_loaders, cmv_data, args)
        (best_acc, best_nmi, best_ari, best_pur), (final_acc, final_nmi, final_ari, final_pur), timing_stats = train(model, optimizer, scheduler, imv_loader, args)

        
        for k in time_record.keys():
            time_record[k].append(timing_stats[k])

        test_record["ACC"].append(final_acc)
        test_record["NMI"].append(final_nmi)
        test_record["ARI"].append(final_ari)
        test_record["PUR"].append(final_pur)
        # lyx-10.30新增
        best_record["ACC"].append(best_acc)
        best_record["NMI"].append(best_nmi)
        best_record["ARI"].append(best_ari)
        best_record["PUR"].append(best_pur)


    print('Average total_train_time {:.3f}s'.format(np.mean(time_record["total_train_time"])))
    print('Average avg_epoch_time   {:.3f}s'.format(np.mean(time_record["avg_epoch_time"])))
    print('Average avg_batch_time   {:.6f}s'.format(np.mean(time_record["avg_batch_time"])))
    print('Average total_eval_time  {:.3f}s'.format(np.mean(time_record["total_eval_time"])))
    print('Average avg_eval_time    {:.3f}s'.format(np.mean(time_record["avg_eval_time"])))
    print('BEST ACC {:.2f} BEST NMI {:.2f} BEST ARI {:.2f} BEST PUR {:.2f}'.format(
        np.max(best_record["ACC"]) * 100,
        np.max(best_record["NMI"]) * 100,
        np.max(best_record["ARI"]) * 100,
        np.max(best_record["PUR"]) * 100))
    print('Average ACC {:.2f} Average NMI {:.2f} Average ARI {:.2f} Average PUR {:.2f}'.format(np.mean(test_record["ACC"]) * 100,
                                                                                               np.mean(test_record["NMI"]) * 100,
                                                                                               np.mean(test_record["ARI"]) * 100,
                                                                                               np.mean(test_record["PUR"]) * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=400, help='training epochs')
    parser.add_argument('--initial_epochs', type=int, default=200, help='initialization epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--prior_learning_rate', type=float, default=0.05, help='initial mixture-of-gaussian learning rate')
    parser.add_argument('--z_dim', type=int, default=10, help='latent dimensions')
    parser.add_argument('--lr_decay_step', type=float, default=10, help='StepLr_Step_size')
    parser.add_argument('--lr_decay_factor', type=float, default=0.9, help='StepLr_Gamma')

    parser.add_argument('--dataset_name', type=str, default="Cifar10")
    parser.add_argument('--test_times', type=int, default=1)
    parser.add_argument('--missing_rate', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=5)
    parser.add_argument('--beta', type=float, default=5)
    parser.add_argument('--temperature', type=float, default=1) 
    parser.add_argument('--temperature_cluster', type=float, default=1)
    parser.add_argument('--temperature_prototype', type=float, default=1) 

    parser.add_argument('--confidence_threshold', type=float, default=0.3, help='threshold for boundary samples') 
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.dataset_dir_base = "./npz_data/"

    if args.dataset_name == "Cifar10":
        args.alpha = 5
        args.beta = 1
        args.seed = 5
        args.epochs = 400
        args.likelihood = 'Gaussian'
    # else:
    #     args.dataset_name = ""
    #     args.alpha = 
    #     args.seed = 
    #     args.likelihood = ''
    #     args.batch_size = 

    for missing_rate in [0.0]:
        args.missing_rate = missing_rate
        print(f"Dataset : {args.dataset_name:<15} Missing rate : {args.missing_rate}")
        test_record = {"ACC": [], "NMI": [], "PUR": [], "ARI": []}
        best_record = {"ACC": [], "NMI": [], "PUR": [], "ARI": []}
        main(args)
