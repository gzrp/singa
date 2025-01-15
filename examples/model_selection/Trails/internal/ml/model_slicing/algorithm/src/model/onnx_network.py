import argparse
from typing import List
import torch
import torch.nn as nn

from copy import deepcopy
from algorithm.src.model.expert import initialize_expert
from algorithm.src.model.gate_network import SparseVerticalGate

class EntmaxBisect_function(nn.Module):
    def __init__(self, alpha=1.5, dim=-1, n_iter=10, requires_grad = True):
        super().__init__()
        self.dim = dim
        self.n_iter = n_iter
        # self.alpha = alpha
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=requires_grad)
        # self.alpha = nn.Parameter(torch.Tensor([alpha]).float(), requires_grad=requires_grad)
    
    def forward(self, X):
        return self._entmax_bisect_(
            X, alpha=self.alpha, dim=self.dim, n_iter=self.n_iter, ensure_sum_one=True
        )
    
    def _entmax_bisect_(self, X, alpha=1.5, dim = -1, n_iter = 10, ensure_sum_one=True):
        
        alpha_shape = list(X.shape)
        alpha_shape[dim] = 1
        alpha = alpha.expand(*alpha_shape)

        d = X.shape[dim]
        
        X = X * (alpha - 1)

        max_val, _ = X.max(dim=dim, keepdim=True)
        tmp = (1 ** (alpha - 1))
        tau_lo = max_val - tmp
        tau_hi = max_val - ((1 / d) ** (alpha - 1))

        f_lo = (torch.clamp(X - tau_lo, min=0) ** (1 / (alpha - 1))).sum(dim) - 1

        dm = tau_hi - tau_lo

        for it in range(n_iter):
            dm /= 2
            tau_m = tau_lo + dm
            p_m = (torch.clamp(X - tau_m, min=0) ** (1 / (alpha - 1)))
            f_m = p_m.sum(dim) - 1

            mask = (f_m * f_lo >= 0).unsqueeze(dim)
            tau_lo = torch.where(mask, tau_m, tau_lo)

        if ensure_sum_one:
            p_m /= p_m.sum(dim=dim).unsqueeze(dim=dim)
        return p_m

    
class ONNXSlicedModel(nn.Module):
    
    def __init__(self, embedding:nn.Module, experts:List[nn.Module], weight:torch.Tensor):
        super().__init__()
        
        self.embedding = embedding
        self.experts = nn.ModuleList(experts)
        self.weight = nn.Parameter(weight) # [1, K]
    
    def forward(self, x):
        """
        x : [B,F]
        """
        x_emb = self.embedding(x)   # [B,F,E]
        x_list = [expert(x_emb) for expert in self.experts]
        x = torch.stack(x_list, dim = 1)    # [B, K, nhid]
        x = self.weight.unsqueeze(-1) * x
        x = torch.sum(x, dim = 1)           # [B, nhid]
        x = x.squeeze(-1)
        
        return torch.sigmoid(x)


class ONNXGeneralModel(nn.Module):

    def __init__(self, args:argparse.Namespace):
        super().__init__()
        
        self.args = args
        
        self.gate_input_size = args.nfield * args.sql_nemb
        self.input_size = args.nfield * args.data_nemb
        self.hidden_size = args.moe_hid_layer_len
        self.output_size = args.output_size
        self.num_experts = args.K
        
        # 
        self.sql_embedding = nn.Embedding(args.nfield + args.nfeat + 1, args.sql_nemb)
        self.input_embedding = nn.Embedding(args.nfeat, args.data_nemb)

        # 
        self.gate_network = SparseVerticalGate(self.gate_input_size, self.num_experts ,args.hid_layer_len, 0.1)
        
        expert = initialize_expert(args)
        
        #
        self.experts = nn.ModuleList([deepcopy(expert) for _ in range(self.num_experts)])
        
        nn.init.xavier_uniform_(self.sql_embedding.weight)
        nn.init.xavier_uniform_(self.input_embedding.weight)
        
        #
        self.sparsemax = EntmaxBisect_function(args.alpha)
    
    
    @torch.no_grad( )
    def forward(self, x:torch.Tensor, sql:torch.Tensor):
        """
        Use sql to customize model and batch process x
        Args: 
        x: [B, nfield] only Id
        sql: [1, nfield] 
        Returns:
        y: [B]
        """
        self.eval() 
        if self.args.expert == "afn":
            self.embedding_clip()
    
        # assert sql.shape[0] == 1 and sql.shape[1] == x.shape[1]
                
        x_emb = self.input_embedding(x)     
        sql_emb = self.sql_embedding(sql)
        # x [B, nfield, data_nemb] sql [1, nfield, sql_nemb]
        
        sql_emb = sql_emb.view(sql_emb.shape[0], -1)
        # flag sql_emb
        
        gate_score = self.sparsemax(self.gate_network(sql_emb))
        # [1, K]
        
        x_list = [expert(x_emb) for expert in self.experts]
        x = torch.stack(x_list, dim = 1 )
        # [B, K, nhid]
        
        x = gate_score.unsqueeze(-1) * x
        # [B, K, nhid]
        
        x = torch.sum(x, dim = 1)
        # [B, nhid]
        return torch.sigmoid(x.squeeze(-1   ))
    
    
    @torch.no_grad()
    # def tailer_by_sql(self, sql:torch.Tensor) -> ONNXSlicedModel:
    def tailer_by_sql(self, sql:torch.Tensor):
        
        """
        sql: [1, F]
        """
        self.eval() 
        
        B = sql.shape[0]
        sql_emb = self.sql_embedding(sql).view(B, -1)
        gate_score = self.gate_network(sql_emb)
        gate_score = self.sparsemax(gate_score)
        
        expert_score = gate_score.squeeze(0)
        non_zero_indices = torch.nonzero(expert_score).t()
        non_zero_index = non_zero_indices.squeeze(0).cpu()
        non_zero_index = non_zero_index.numpy().tolist()
        
        non_zero_scores = expert_score[non_zero_indices] #[B,K], K-> none zero scores
        
        selected_experts = [ deepcopy(self.experts[i]) for i in non_zero_index]
        embedding = deepcopy(self.input_embedding)
        
        print("sliced expert number:{}".format(len(non_zero_index)))
        
        return ONNXSlicedModel(embedding, selected_experts, non_zero_scores)
        
    

        

        