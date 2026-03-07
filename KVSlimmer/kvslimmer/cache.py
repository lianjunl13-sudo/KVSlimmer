import math
import torch

from asymkv.streaming_llm.kv_cache import StartRecentKVCache

from kvslimmer.utils import repeat_kv_3d
from kvslimmer.merge import optimal_merge_k_from_alpha_d


class KVSlimmerCache(StartRecentKVCache):
   
    def formalize_past_key_values(self, past_key_values):
        ret = []
        for layer in past_key_values:
            if len(layer) == 3:
                ret.append(layer)
                continue

            k, v = layer
            l = torch.ones(
                k.size()[:-1],
                device=k.device,
                dtype=k.dtype,
            )
            ret.append((k, v, l))
        return tuple(ret)

    def __call__(
        self,
        past_key_values,
        attns,
        hessian_diagonal=None,  
    ):
        if past_key_values is None:
            return None

        if len(past_key_values[0]) == 2:
            past_key_values = self.formalize_past_key_values(past_key_values)

        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values, hessian_diagonal

       
        new_mid = []
        new_hessian_mid = []

        for layer_idx, (k, v, l) in enumerate(past_key_values):
            mid_k = self.k_slice(k, self.start_size, seq_len)   # (B, kvH, Tmid, D)
            mid_v = self.v_slice(v, self.start_size, seq_len)   # (B, kvH, Tmid, Dv)
            mid_l = l[:, :, self.start_size:seq_len]            # (B, kvH, Tmid)

            B, kvH, Tmid, Dk = mid_k.shape
            Tpair = max(0, Tmid - 1)

            if Tpair == 0:
                new_mid.append((mid_k, mid_v, mid_l))

                if hessian_diagonal is not None:
                    hk_mid_4d = self.k_slice(hessian_diagonal[layer_idx][0], self.start_size, seq_len)
                    new_hessian_mid.append((hk_mid_4d, None))
                else:
                    new_hessian_mid.append((None, None))
                continue

          
            attn_raw = attns[layer_idx][:, :, :, self.start_size:]    
            attn_pos = attn_raw.sum(dim=-2)                           

            rep_attn = int(attn_pos.shape[1] / mid_l.shape[1])      
            rep_mid_l = repeat_kv_3d(mid_l, rep_attn)                  
            attn_pos = attn_pos * rep_mid_l[:, :, :attn_pos.shape[2]]  

            attn_pair = attn_pos[:, :, :-1] + attn_pos[:, :, 1:]      

            rep = attn_pair.shape[1] // mid_v.shape[1]                
            attn_pair_kv = attn_pair.view(B, kvH, rep, -1).mean(dim=2) 

        
            weight_i = attn_pair_kv.sum(dim=1).squeeze(0)            

            l_i = mid_l.sum(dim=-2)                                   
            l_i = l_i[:, :-1] + l_i[:, 1:]                  
            l_i = l_i[0].sum(dim=0)                                

            gamma = 4096 / math.log(512)
            weight_idx = torch.arange(
                1,
                len(weight_i) + 1,
                device=weight_i.device,
                dtype=weight_i.dtype,
            )
            sqrt_indices = torch.exp(weight_idx / gamma)

            weight_i = weight_i / sqrt_indices * l_i

            K = max(0, min(seq_len - self.cache_size, Tpair))
            if K == 0:
                new_mid.append((mid_k, mid_v, mid_l))

                if hessian_diagonal is not None:
                    hk_mid_4d = self.k_slice(hessian_diagonal[layer_idx][0], self.start_size, seq_len)
                    new_hessian_mid.append((hk_mid_4d, None))
                else:
                    new_hessian_mid.append((None, None))
                continue

            mink_indices = weight_i.topk(K, dim=-1, largest=False).indices  # (K,)

            eps = 1e-21
            k1 = mid_k[:, :, :-1, :]
            k2 = mid_k[:, :, 1:, :]

            if hessian_diagonal is None:
                raise ValueError("hessian_diagonal  must be provided.")

         
            hk_mid_4d = self.k_slice(hessian_diagonal[layer_idx][0], self.start_size, seq_len)
            h_mid = hk_mid_4d[..., 0]

            attn_pos_kv = attn_pos.view(B, kvH, rep_attn, -1).mean(dim=2)  # (B, kvH, Tmid)
            alpha_tok = attn_pos_kv / (attn_pos_kv.sum(dim=-1, keepdim=True) + eps)

            alpha1 = alpha_tok[:, :, :-1]
            alpha2 = alpha_tok[:, :, 1:]

           
            denom = alpha_tok * (1.0 - 2.0 * alpha_tok) 
            d_tok = h_mid / (denom + eps)                # (B, kvH, Tmid)

            d1 = d_tok[:, :, :-1]
            d2 = d_tok[:, :, 1:]

            h1 = h_mid[:, :, :-1]
            h2 = h_mid[:, :, 1:]

            a, b, ke, h12 = optimal_merge_k_from_alpha_d(
                k1=k1,
                k2=k2,
                alpha1=alpha1,
                alpha2=alpha2,
                d1=d1,
                d2=d2,
                eps=eps,
                h11=h1,
                h22=h2,
            )

            ve = mid_v[:, :, :-1, :] + mid_v[:, :, 1:, :]
            le = mid_l[:, :, :-1] + mid_l[:, :, 1:]

            h_merge = a**2 * h1 + b**2 * h2  # (B, kvH, Tmid-1)

            mask = torch.ones(mid_k.shape[2], dtype=torch.bool, device=mid_k.device)
            mask[mink_indices + 1] = False

            mid_k[:, :, mink_indices, :] = ke[:, :, mink_indices, :]
            mid_v[:, :, mink_indices, :] = ve[:, :, mink_indices, :]
            mid_l[:, :, mink_indices] = le[:, :, mink_indices]

            h_mid[:, :, mink_indices] = h_merge[:, :, mink_indices]
            new_h_mid = h_mid[:, :, mask]  # (B, kvH, Tnew)

            new_mid_k = mid_k[:, :, mask, :]
            new_mid_v = mid_v[:, :, mask, :]
            new_mid_l = mid_l[:, :, mask]
            new_mid_l = torch.clamp(new_mid_l, max=5)

            new_mid.append((new_mid_k, new_mid_v, new_mid_l))

            new_h_mid_4d = new_h_mid[..., None].expand(
                B, kvH, new_h_mid.shape[-1], Dk
            ).contiguous()
            new_hessian_mid.append((new_h_mid_4d, None))


        new_past = [
            [
                torch.cat([self.k_slice(k, 0, self.start_size), new_k], dim=self.k_seq_dim),
                torch.cat([self.v_slice(v, 0, self.start_size), new_v], dim=self.v_seq_dim),
                torch.cat([l[:, :, : self.start_size], new_l], dim=2),
            ]
            for (k, v, l), (new_k, new_v, new_l) in zip(past_key_values, new_mid)
        ]

        if hessian_diagonal is None:
            return new_past, None

        new_hess = []
        for i, (new_h_mid_4d, _) in enumerate(new_hessian_mid):
            if new_h_mid_4d is None:
                new_hess.append((None, None))
                continue

            h_start = hessian_diagonal[i][0][:, :, : self.start_size, :]
            h_new = torch.cat([h_start, new_h_mid_4d], dim=2)
            new_hess.append((h_new, None))

        return new_past, new_hess