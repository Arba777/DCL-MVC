import torch
import numpy as np
import torch.nn.functional as F


def log_gaussian(x, mu, var): 
    return -0.5 * (torch.log(torch.tensor(2.0 * np.pi)) + torch.log(var) + torch.pow(x - mu, 2) / var)


def gaussian_kl(q_mu, q_var, p_mu, p_var):  
    return - 0.5 * (torch.log(q_var / p_var) - q_var / p_var - torch.pow(q_mu - p_mu, 2) / p_var + 1)


def vade_trick(mc_sample, mog_pi, mog_mu, mog_var):
    log_pz_c = torch.sum(log_gaussian(mc_sample.unsqueeze(1), mog_mu.unsqueeze(0), mog_var.unsqueeze(0)), dim=-1) # log p(z|c)
    log_pc = torch.log(mog_pi.unsqueeze(0)) 
    log_pc_z = log_pc + log_pz_c 
    pc_z = torch.exp(log_pc_z) + 1e-10
    normalized_pc_z = pc_z / torch.sum(pc_z, dim=1, keepdim=True)
    return normalized_pc_z 


def kl_term(z_mu, z_var, qc_x, mog_weight, mog_mu, mog_var):
    z_kl_div = torch.sum(qc_x * torch.sum(gaussian_kl(z_mu.unsqueeze(1), z_var.unsqueeze(1), mog_mu.unsqueeze(0), mog_var.unsqueeze(0)), dim=-1),
                         dim=1)
    z_kl_div_mean = torch.mean(z_kl_div)

    c_kl_div = torch.sum(qc_x * torch.log(qc_x / mog_weight.unsqueeze(0)), dim=1)
    c_kl_div_mean = torch.mean(c_kl_div)
    return z_kl_div_mean, c_kl_div_mean 

def coherence_function(vs_mus, vs_vars, aggregated_mu, aggregated_var, mask=None):
    coherence_loss_list = []
    mask_stack = torch.cat(mask, dim=1)  # Batch size * V
    norm = torch.sum(mask_stack, dim=1)  # |Via|
    for v in range(len(vs_mus)):
        uniview_coherence_loss = torch.sum(gaussian_kl(aggregated_mu, aggregated_var, vs_mus[v], vs_vars[v]), dim=1)
        exist_loss = uniview_coherence_loss * mask[v].squeeze()
        coherence_loss_list.append(exist_loss)
    coherence_loss = torch.mean(sum(coherence_loss_list) / norm)
    return coherence_loss 

def coherence_function_att(vs_mus, vs_vars, aggregated_mu, aggregated_var, alphas, mask=None, eps=1e-8):
    """
    Attention-weighted coherence loss:  L_CH_att = Σ_v α_v · KL( q_att || q_v )
    Returns a scalar tensor.
    """

    if mask is not None:
        mask_stack = torch.cat(mask, dim=1).float()        # (B, V)
        alphas = alphas * mask_stack                       # zero-out missing views
        alphas = alphas / (alphas.sum(dim=1, keepdim=True) + eps)  # re-normalize
    else:
        
        alphas = alphas / (alphas.sum(dim=1, keepdim=True) + eps)

    
    kl_list = []
    for mu_v, var_v in zip(vs_mus, vs_vars):
        
        kl = torch.sum(gaussian_kl(aggregated_mu, aggregated_var, mu_v, var_v), dim=1)  # (B,)
        kl_list.append(kl)
    kl_stack = torch.stack(kl_list, dim=1)  # (B, V)

    
    weighted_kl = (alphas * kl_stack).sum(dim=1)           # (B,)
    coherence_loss = weighted_kl.mean()                    # scalar

    return coherence_loss

def contrastiveLoss(H, r, mask, temperature, weight=None, eps=1e-8):
    device = H.device
    H_norm = F.normalize(H, dim=1)
    R_norm = F.normalize(r, dim=1)
    sim_mat = torch.matmul(H_norm, R_norm.T) / temperature

    mask = mask.view(-1).bool()
    idx = torch.nonzero(mask).squeeze()

    if idx.numel() < 2:
        return torch.tensor(0., device=device, requires_grad=True)

    sim_valid = sim_mat[idx][:, idx]  # [M, M]
    if not torch.isfinite(sim_valid).all():
        return torch.tensor(0., device=device, requires_grad=True)

    logits = sim_valid - sim_valid.max(dim=1, keepdim=True)[0]
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + eps)

    loss = -log_prob[torch.arange(log_prob.size(0)), torch.arange(log_prob.size(0))]
    loss = loss.mean()
    if weight is not None:
        loss = loss * weight
    return loss


def cluster_contrastive_loss(mog_mu, temperature=1.0):
    K = mog_mu.shape[0]
    
    cluster_contractive_norm = F.normalize(mog_mu, dim=1) # K * z_dim

    
    similarity = torch.matmul(cluster_contractive_norm, cluster_contractive_norm.t()) / temperature

    
    labels = torch.arange(K, device = mog_mu.device) 

    
    loss = F.cross_entropy(similarity, labels)
    return loss


def prototype_contrastive_loss(z, qc_x, mog_mu, temperature=1.0, confidence_threshold=0.3, epoch=None, total_epochs=None):
    N, K = qc_x.shape

    
    if epoch is None:
        stage = "early"
    elif epoch < total_epochs * 0.3:
        stage = "early"
    elif epoch < total_epochs * 0.7:
        stage = "middle"
    else:
        stage = "late"


    sorted_qc_x, _ = torch.sort(qc_x, dim=1, descending=True)
    confidence = sorted_qc_x[:, 0] - sorted_qc_x[:, 1]  
    

    boundary_mask = confidence < confidence_threshold

    z_norm = F.normalize(z, dim=1)
    prototypes_norm = F.normalize(mog_mu, dim=1)

    similarity = torch.matmul(z_norm, prototypes_norm.T) / temperature

    if stage == "early":

        weights = qc_x  # [N, K]
        

        logits = similarity - similarity.max(dim=1, keepdim=True)[0]  
        exp_logits = torch.exp(logits)
        

        soft_targets = weights / weights.sum(dim=1, keepdim=True)
        loss_per_sample = -torch.sum(soft_targets * F.log_softmax(logits, dim=1), dim=1)
        
    elif stage == "middle":

        loss_per_sample = torch.zeros(N, device=z.device)
        

        high_conf_mask = ~boundary_mask
        if high_conf_mask.sum() > 0:
            high_conf_labels = torch.argmax(qc_x[high_conf_mask], dim=1)  # [M]
            high_conf_loss = F.cross_entropy(
                similarity[high_conf_mask], 
                high_conf_labels, 
                reduction='none'
            )
            loss_per_sample[high_conf_mask] = high_conf_loss

        if boundary_mask.sum() > 0:
            boundary_weights = qc_x[boundary_mask]
            boundary_logits = similarity[boundary_mask]
            
            boundary_logits = boundary_logits - boundary_logits.max(dim=1, keepdim=True)[0]
            boundary_soft_targets = boundary_weights / boundary_weights.sum(dim=1, keepdim=True)
            boundary_loss = -torch.sum(
                boundary_soft_targets * F.log_softmax(boundary_logits, dim=1), 
                dim=1
            )
            loss_per_sample[boundary_mask] = boundary_loss
    
    else:  # stage == "late"

        loss_per_sample = torch.zeros(N, device=z.device)
        

        certain_mask = confidence >= (confidence_threshold * 0.5) 
        if certain_mask.sum() > 0:
            certain_labels = torch.argmax(qc_x[certain_mask], dim=1)
            certain_loss = F.cross_entropy(
                similarity[certain_mask], 
                certain_labels, 
                reduction='none'
            )
            loss_per_sample[certain_mask] = certain_loss
        

        very_uncertain_mask = ~certain_mask
        if very_uncertain_mask.sum() > 0:
            uncertain_weights = qc_x[very_uncertain_mask]
            uncertain_logits = similarity[very_uncertain_mask]
            
            uncertain_logits = uncertain_logits - uncertain_logits.max(dim=1, keepdim=True)[0]
            uncertain_soft_targets = uncertain_weights / uncertain_weights.sum(dim=1, keepdim=True)
            uncertain_loss = -torch.sum(
                uncertain_soft_targets * F.log_softmax(uncertain_logits, dim=1), 
                dim=1
            )
            loss_per_sample[very_uncertain_mask] = uncertain_loss
    
    return loss_per_sample.mean()
    

def get_contrastive_weights(epoch, total_epochs):

    if epoch < total_epochs * 0.3:  # early
        return 8
    elif epoch < total_epochs * 0.7:  # mid
        return 5
    else:  # late
        return 1

