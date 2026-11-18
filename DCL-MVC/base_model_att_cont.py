import torch
import torch.nn as nn
import math


class Gaussian_sampling(nn.Module):
    def forward(self, mu, var):
        std = torch.sqrt(var)
        # epi = std.data.new(std.size()).normal_()
        epi = torch.randn_like(std)
        return epi * std + mu 


class view_specific_encoder(nn.Module): 
    def __init__(self, view_dim, latent_dim):
        super(view_specific_encoder, self).__init__()
        self.x_dim = view_dim 
        self.z_dim = latent_dim 
        self.hidden_dim = 2000 
        self.encoder = nn.Sequential(nn.Linear(self.x_dim, 500),
                                     nn.ReLU(),
                                     nn.Linear(500, 500),
                                     nn.ReLU(),
                                     nn.Linear(500, self.hidden_dim),
                                     nn.ReLU()
                                     )
        self.z_mu = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_var = nn.Sequential(nn.Linear(self.hidden_dim, self.z_dim), nn.Softplus())

    def forward(self, x):
        hidden_feature = self.encoder(x)
        vs_mu = self.z_mu(hidden_feature)
        vs_var = self.z_var(hidden_feature)
        return hidden_feature, vs_mu, vs_var


class view_specific_decoder(nn.Module):
    def __init__(self, view_dim, latent_dim):
        super(view_specific_decoder, self).__init__()
        self.x_dim = view_dim
        self.z_dim = latent_dim
        self.hidden_dim = 2000
        self.decoder = nn.Sequential(nn.Linear(self.z_dim, self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, 500),
                                     nn.ReLU(),
                                     nn.Linear(500, 500),
                                     nn.ReLU(),
                                     nn.Linear(500, self.x_dim),
                                     )

    def forward(self, z):
        xr = self.decoder(z)
        return xr

class SingleSelfAttention(nn.Module):
    def __init__(self, d_in, d_k = None):
        super().__init__()
        d_k = d_k or d_in
        self.q_proj = nn.Linear(d_in, d_k, bias=False)
        self.k_proj = nn.Linear(d_in, d_k, bias=False)
        self.v_proj = nn.Linear(d_in, d_in, bias=False)
        self.norm = nn.LayerNorm(d_in)
        self.scale = math.sqrt(d_k)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, hidden, mask=None):
        residual = hidden.clone()
        Q = self.q_proj(hidden)
        K = self.k_proj(hidden)
        V = self.v_proj(hidden)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale # (B, V, V)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1) # (B, V, V)
        out = torch.matmul(weights, V) # (B, V, D)
        out = self.dropout(out)
        out = self.norm(out + residual)
        return out, weights
        


class DVIMC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.x_dim_list = args.multiview_dims 
        self.k = args.class_num 
        
        self.z_dim = args.z_dim
        self.num_views = args.num_views 

        self.prior_weight = nn.Parameter(torch.full((self.k,), 1 / self.k), requires_grad=True)
        self.prior_mu = nn.Parameter(torch.full((self.k, self.z_dim), 0.0), requires_grad=True)
        self.prior_var = nn.Parameter(torch.full((self.k, self.z_dim), 1.0), requires_grad=True)

        self.encoders = nn.ModuleDict({f'view_{v}': view_specific_encoder(self.x_dim_list[v], self.z_dim) for v in range(self.num_views)})
        self.decoders = nn.ModuleDict({f'view_{v}': view_specific_decoder(self.x_dim_list[v], self.z_dim) for v in range(self.num_views)})

        hidden_dim = self.encoders['view_0'].hidden_dim
        self.attention = SingleSelfAttention(d_in=hidden_dim)

        self.common_information_module = nn.ModuleDict({f'view_{v}':nn.Sequential(nn.Linear(hidden_dim, hidden_dim)) for v in range(self.num_views)})

        self.cross_mu = nn.Linear(hidden_dim, self.z_dim)
        self.cross_var = nn.Sequential(nn.Linear(hidden_dim, self.z_dim), nn.Softplus())

        self.sampling_fn = Gaussian_sampling()

    def mv_encode(self, x_list):
        latent_representation_list = []
        for v in range(self.num_views):
            _, latent_representation, _ = self.encoders[f'view_{v}'](x_list[v])
            latent_representation_list.append(latent_representation)
        return latent_representation_list

    def sv_encode(self, x, view_idx):
        _, latent_representation, _ = self.encoders[f'view_{view_idx}'](x)
        xr = self.decoders[f'view_{view_idx}'](latent_representation)
        return latent_representation, xr
    
    def mv_hidden(self, x_list):
        h_list, vs_mus, vs_vars = [], [], []
        for v in range(self.num_views):
            h, vs_mu, vs_var = self.encoders[f'view_{v}'](x_list[v])
            h_list.append(h)
            vs_mus.append(vs_mu)
            vs_vars.append(vs_var)
        return h_list, vs_mus, vs_vars

    def inference_z(self, x_list, mask=None):
        # extract hidden features from each view
        hidden_list, vs_mus, vs_vars = self.mv_hidden(x_list)
        if mask is not None:
            mask_tensor = torch.cat(mask, dim=1)
            # mask: (batch, V) BoolTensor
            for v in range(self.num_views):
                hidden_list[v] = hidden_list[v] * mask_tensor[:, v].unsqueeze(1)
        else:
            mask_tensor = None

        H_stack = torch.stack(hidden_list, dim=1)

        H_attn, attn_weights = self.attention(H_stack, mask_tensor.bool())
        if torch.isnan(H_attn).any():
            print(H_attn)
            raise RuntimeError("H_att输出 NaN")
        w = attn_weights.mean(dim=1)
        w = w / w.sum(dim=1, keepdim=True)

        r_list = [self.common_information_module[f'view_{v}'](hidden_list[v]) for v in range(self.num_views)]
       
        if mask is not None:
            valid_counts = mask_tensor.sum(dim=1, keepdim=True).clamp(min=1).to(H_attn.dtype)  
            H_fused = H_attn.sum(dim=1) / valid_counts
        else:
            H_fused = H_attn.mean(dim=1)
        
        mu_cross = self.cross_mu(H_fused)
        var_cross = self.cross_var(H_fused)
        return vs_mus, vs_vars, mu_cross, var_cross, w, r_list, H_fused


    def generation_x(self, z):
        xr_list = [vs_decoder(z) for vs_decoder in self.decoders.values()]
        return xr_list

    def forward(self, x_list, mask=None):
        vs_mus, vs_vars, aggregated_mu, aggregated_var, att_weight, R, H_fused = self.inference_z(x_list, mask)
        if torch.isnan(H_fused).any():
            raise RuntimeError("H: NaN")

        z_sample = self.sampling_fn(aggregated_mu, aggregated_var)
        xr_list = self.generation_x(z_sample)
        vade_z_sample = self.sampling_fn(aggregated_mu, aggregated_var) 
        return vs_mus, vs_vars, aggregated_mu, aggregated_var, xr_list, vade_z_sample, att_weight, R, H_fused
