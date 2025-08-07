import torch
import triton
import triton.language as tl
import math


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,   #Bq
    K_TILE_SIZE: tl.constexpr,   #Bk
    is_casual: tl.constexpr
):
    
    # program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0)
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0)
    )

    v_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0,0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0)
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob + query_tile_index * Q_TILE_SIZE * stride_oq,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb + query_tile_index * Q_TILE_SIZE * stride_lq,
        shape = (N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    m = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    Tk = (N_KEYS-1) // K_TILE_SIZE + 1
    for j in range(Tk):
        q = tl.load(Q_block_ptr)     #(16, 64)
        k = tl.load(K_block_ptr)       
        v = tl.load(v_block_ptr)

        # tl.device_print('q', q)
        # tl.device_print('shape', q.shape)
        k_transposed = tl.trans(k)
        s = tl.dot(q, k_transposed) * scale        #(16, 16)

        # add casual_mask
        if is_casual:
            query_start = query_tile_index * Q_TILE_SIZE
            key_start = j * K_TILE_SIZE

            q_indices = tl.arange(0, Q_TILE_SIZE) + query_start
            k_indices = tl.arange(0, K_TILE_SIZE) + key_start

            mask = (k_indices[None, :] > q_indices[:,None])
            s = s - mask * 1e6

        # 更新 m, l, o
        m_prev = m
        l_prev = l
        m = tl.maximum(m_prev, tl.max(s, axis=-1))      #(16,)   
        p = tl.exp(s - m[:,None]).to(v_block_ptr.type.element_ty)       #(16,16)
        exp_m_diff = tl.exp(m_prev - m).to(l.dtype)     #(16,)
        l = exp_m_diff * l_prev + tl.sum(p, axis=-1)    # (16,)
        o = exp_m_diff[:,None] * o + tl.dot(p, v)       # (16, 64)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        v_block_ptr = v_block_ptr.advance((K_TILE_SIZE, 0))
    
    o_normalized = o / l[:,None]
    output_dtype = O_block_ptr.type.element_ty
    tl.store(O_block_ptr, o_normalized.to(output_dtype))
    L_i = (m + tl.log(l)).to(L_block_ptr.type.element_ty)
    tl.store(L_block_ptr, L_i)


class FlashAttention2_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_casual=False):
        batch_size, Nq, D = Q.shape
        _, Nk, _  = K.shape
        assert K.shape == V.shape, "Dimension mismatch"
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()

        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        Tq = (Nq - 1) // Q_TILE_SIZE + 1
        Tk = (Nk - 1) // K_TILE_SIZE + 1

        # allocate output tensors
        O = torch.empty_like(Q)
        L = torch.empty((batch_size, Nq), device=Q.device, dtype=torch.float32)

        scale = 1 / math.sqrt(D)

        grid = (Tq, batch_size)

        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),

            Nq, Nk,
            scale,
            D,
            Q_TILE_SIZE, 
            K_TILE_SIZE,
            is_casual = is_casual
        )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_casual = is_casual
        
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








        


        









