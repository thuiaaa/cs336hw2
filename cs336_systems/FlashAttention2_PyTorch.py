import torch
import pdb
import typing


class FlashAttention2_PyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_casual=False):
        # import pdb; pdb.set_trace()
        device = Q.device
        # Define output
        O = torch.zeros(Q.shape, device=device)
        L = torch.zeros(Q.shape[:-1],device=device)

        Nq, d = Q.shape[1], Q.shape[-1]
        Nk = K.shape[1]

        # set Bq, Bk
        Bq = 16
        Bk = 16

        Tq = (Nq - 1) // Bq + 1 # round to ceil
        Tk = (Nk - 1) // Bk + 1

        O_list = [[torch.zeros((Bq, d), device=device) for _ in range(Tk+1)] for _ in range(Tq+1)]
        l_list = [[torch.zeros((Bq, ), device=device) for _ in range(Tk+1)] for _ in range(Tq+1)]
        m_list = [[torch.full((Bq, ), float('-inf'), device=device, ) for _ in range(Tk+1)] for _ in range(Tq+1)]
        S_list = [[None for _ in range(Tk+1)] for _ in range(Tq+1)]
        P_list = [[None for _ in range(Tk+1)] for _ in range(Tq+1)]
        for i in range(1, Tq+1):
            Q_start, Q_end = (i-1) * Bq, i * Bq
            Qi = Q[:, Q_start:Q_end, :]
            for j in range(1, Tk+1):
                K_start, K_end = (j-1) * Bk, j * Bk
                Kj = K[:, K_start:K_end, :]
                Vj = V[:, K_start:K_end, :]
                S_list[i][j] = torch.matmul(Qi, Kj.transpose(-1,-2)) / (d ** 0.5)  #(Bq,Bk)
                m_list[i][j] = torch.max(m_list[i][j-1], torch.max(S_list[i][j], dim=-1).values)   # (Bq,)
                P_list[i][j] = torch.exp(S_list[i][j] - m_list[i][j].unsqueeze(-1))    #(Bq, Bk)
                l_list[i][j] = torch.exp(m_list[i][j-1]-m_list[i][j]) * l_list[i][j-1].unsqueeze(0) + torch.sum(P_list[i][j], dim=-1) #(Bq,)
                O_list[i][j] = torch.matmul(torch.diag_embed(torch.exp(m_list[i][j-1]-m_list[i][j])) , O_list[i][j-1]) + torch.matmul( P_list[i][j] , Vj)
            Oi = torch.matmul(torch.diag_embed(1 /l_list[i][Tk]), O_list[i][Tk])
            print(Oi)
            Li = m_list[i][Tk] + torch.log(l_list[i][Tk])
            O[:, (i-1) * Bq:i * Bq, :] = Oi
            L[:, (i-1) * Bq:i * Bq] = Li
            ctx.save_for_backward(L, Q, K, V, O)
            ctx.is_casual = is_casual
        # import pdb; pdb.set_trace()
        return O
    
    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        is_casual = ctx.is_casual

        dim_k = K.shape[-1]
        scale =  dim_k ** 0.5 
        S = torch.matmul(Q, K.transpose(-1,-2)) / scale
        
        if is_casual:
            n_queries = S.shape[-2]
            n_keys = S.shape[-1]
            casual_mask = torch.tril(torch.ones(n_queries, n_keys, dtype=torch.bool, device=S.device))
            S = torch.where(casual_mask, S, -1e6)

        P = torch.exp(S - L.unsqueeze(-1))

        dV = torch.matmul(P.transpose(-1,-2), dO)
        dP = torch.matmul(dO, V.transpose(-1,-2))
        D = torch.sum((O * dO), dim=-1)
        dS = P * (dP - D.unsqueeze(-1))
        dQ = torch.matmul(dS, K) / scale
        dK = torch.matmul(dS.transpose(-1,-2), Q) / scale
        return dQ, dK, dV, None









