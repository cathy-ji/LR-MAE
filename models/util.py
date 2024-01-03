import numpy as np
import torch
import time
from itertools import chain
def swap_pair_n(data, attn):
    """
    swap batch of "input data" based on the attention, with n_div (last attention dimension size) division.
    data: list of data, can be [data1] only or [data1,data2]
        data1 : [BS, n_pts, n_feature or emb_dim] -> data that will be mixed up based on attention division
        data2: [BS, n_pts, n_feature or emb_dim] -> another data that will be mixed up, use case: for 'original_feature'
                                                    before attention block, that will be used in residual attention addition ops.
    attn : [BS, n_pts, n_div]
    return:
        new_data: [BS, n_pts, emb_dim]
    """
    data = [data]
    bs, n_pts, n_col = data[0].size()
    n_div = attn.size(-1)
    if bs%n_div != 0: raise Exception("Number of sample must be divisible by n_div!")
    n_pair = (int)(bs / n_div)
    #define the composition of new samples
    idx_perm = np.random.permutation(n_div)
    comp_idx = np.stack([np.roll(idx_perm,shift=i) for i in range(n_div)], axis=0)#(n_div,n_div)
    attn = torch.argmax(attn,dim=-1) #[BS, n_pts]
    attn = attn.detach().cpu().numpy()
    idx_helper_list = []
    # if is_visualize:
    #     class_map = get_class_mapping_modelnet40()
    #     path = f"{cfg.EXP.WORKING_DIR}/{args.model_path}/{dir}"
    #     if not os.path.exists(path):
    #         os.makedirs(path)

    # build new sample indices
    t0 = time.time()
    new_sample_indices = []
    for pair_idx in range(n_pair):
        idx_helper = [sample_pair_idx*n_pair+pair_idx for sample_pair_idx in range(n_div)]#(n_vid ) 属于同一类的n_div个样本的idx
        temp_attn = np.stack([(attn[sample_pair_idx*n_pair+pair_idx]) for sample_pair_idx in range(n_div)], axis=0) #[n_div,n_point] 属于同一个类的attn
        for sample_pair_idx in range(n_div):
            ## compose a new sample indices from every samples in the pair
            temp_list = list(chain(*[list(np.argwhere(temp_attn[i]==comp_idx[sample_pair_idx][i]).reshape((-1))+(i*n_pts))
                                     for i in range(n_div)]))
            if len(temp_list) < 50: #if number of new sample indices is under threshold, use the original instead!
                temp_list = np.arange(n_pts).tolist()

            ## sample n_pts and resuffle the points order
            n_pts_temp_list = len(temp_list)
            if n_pts_temp_list >= n_pts:  # if the new sample has more points than or == n_pts
                idc = np.random.choice(temp_list, n_pts, replace=False)
            else:
                delta = n_pts - n_pts_temp_list
                idc_add = np.random.choice(temp_list, delta, replace=True if delta>n_pts_temp_list else False)
                idc = np.concatenate((np.asarray(temp_list), idc_add), axis=0)
                np.random.shuffle(idc)
            new_sample_indices.append(idc) #[BS as list, n_pts as 1D array] 存储了同属与一个类的点的下标

        idx_helper_list.append(idx_helper) #[n_pair as list, n_div as list]

    # define re-ordering new sample indices based on original input data
    idx_helper_list = list(chain(*idx_helper_list)) #[BS as list] 每n_div个一组，共有n_pair对，展开成为一维的list
    idx_helper_list_invert = np.argsort(idx_helper_list) #[BS as 1D array]

    # gather new samples based on the constructed indices
    result = []
    new_sample_indices = np.stack(new_sample_indices, axis=0) # [BS, n_pts] --> indices of new samples
    ## re-oder based on the invert indices and gathered tensor
    new_sample_indices = new_sample_indices[idx_helper_list_invert] # [BS, n_pts] --> indices of new samples
    new_sample_indices = torch.from_numpy(new_sample_indices).cuda().unsqueeze(dim=2).repeat(1,1,n_col) #[BS, n_pts=1024, n_col]
    ## create helper indices for gathered data
    idx_helper_list = np.reshape(np.asarray(idx_helper_list),(n_pair,n_div)) #[n_pair, n_div]
    repeat_indices = [np.full((n_div),i) for i in range(n_pair)] #[n_pair, n_div] 
    repeat_indices = np.asarray(repeat_indices).reshape((-1)) #[BS]
    idx_helper_list = idx_helper_list[repeat_indices] #[BS, n_div]
    for data_idx in range (len(data)): #repeat to all data in the data list
        temp_data = torch.cat([data[data_idx][idx_helper_list[:,i]] for i in range(n_div)], dim=1) # [BS, n_pts*n_div, n_col]
        ## re-oder based on the invert indices
        temp_data = temp_data[idx_helper_list_invert] # [BS, n_pts*n_div, n_col]
        ## gather the new samples
        result.append(torch.gather(temp_data,dim=1,index=new_sample_indices)) # list of [BS, n_pts, n_col]
        # if is_visualize:
        #     attn = torch.from_numpy(attn)
        #     temp_attn = torch.cat([attn[idx_helper_list[:,i]] for i in range(n_div)], dim=1)
        #     temp_attn = temp_attn[idx_helper_list_invert]
        #     temp_attn = torch.gather(temp_attn, dim=1, index=new_sample_indices[:,:,0].cpu())

    

    return result[0]