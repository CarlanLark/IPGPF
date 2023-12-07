"""
The codes are based on https://github.com/HangYang-NLP/DE-PPN
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cost_event_type = config.cost_weight["event_type"]
        self.cost_role = config.cost_weight["role"]
        self.num_event_type = config.event_type_classes

    def forward(self, outputs, targets, event_mask_tensor, inf = 1000000):
        num_sets, num_roles, num_entities = outputs["pred_role"].size()
        pred_event = outputs["pred_doc_event"] # [num_sets, num_event_types]
        gold_event = targets["doc_event_label"]
        # gold_event_list = [gold_event_type for gold_event_type in gold_event if gold_event_type != self.num_event_type]
        gold_event_tensor = torch.tensor(gold_event).cuda()
        if self.num_event_type == 2:
            gold_event_tensor = torch.zeros(gold_event_tensor.size()).long().cuda()

        pred_role = outputs["pred_role"]  # [num_sets,num_roles,num_etities]
        gold_role = targets["role_label"]

        gold_role_lists = [role_list for role_list in gold_role if role_list is not None]
        # gold_roles_list = [role for role in gold_role_lists]
        gold_role = torch.tensor(gold_role_lists).cuda()

        pred_role_list = pred_role.split(1,1)
        gold_role_list = gold_role.split(1,1)

        cost_list = []
        for pred_role_tensor, gold_role_tensor in zip(pred_role_list, gold_role_list):
            pred_role_tensor = pred_role_tensor.squeeze(1)
            cost_list.append( -self.cost_role * pred_role_tensor[:, gold_role_tensor.squeeze(1)] )

        event_type_cost = -self.cost_event_type * pred_event[:, gold_event_tensor]

        role_cost_tensor = torch.stack(cost_list)
        role_cost_tensor = role_cost_tensor.transpose(1,0)
        role_cost_tensor = role_cost_tensor.view(num_sets, num_roles, -1)
        role_cost = torch.sum(role_cost_tensor, dim=1)
        all_cost = role_cost + event_type_cost


        indices = linear_sum_assignment(all_cost.cpu().detach().numpy())
        indices_tensor = torch.as_tensor(np.array(indices), dtype=torch.int64)
        row_ind, col_ind = indices_tensor[0], indices_tensor[1]
        last_iter_ind = num_sets - 1
        
        # two-stage matching
        # if there exist non-trained GTs
        if (event_mask_tensor == 0).int().sum().item() > 0:
            last_matching_ind = torch.min(all_cost[last_iter_ind] + event_mask_tensor, dim = 0)[1].item()
            event_mask_tensor[last_matching_ind] = inf 
            last_src_ind, last_tgt_ind = last_iter_ind, last_matching_ind
        # if all GTs have been trained
        else:
            # if last iter generation aligned to golden label
            if (row_ind == last_iter_ind).int().sum().item() == 1:
                last_matching_ind = torch.nonzero(row_ind == last_iter_ind).sum().item() # get last iter matching index
                last_src_ind, last_tgt_ind = row_ind[last_matching_ind].item(), col_ind[last_matching_ind].item() # ind in data score
            # if last iter generation aligned to None, greedy align it to the most similar GT
            elif (row_ind == last_iter_ind).int().sum().item() == 0:
                last_matching_ind = torch.min(all_cost[last_iter_ind], dim = 0)[1].item()
                last_src_ind, last_tgt_ind = last_iter_ind, last_matching_ind
            else:
                raise Exception('Error: last iter generation matching error')
        #last_src_ind, last_tgt_ind, will be tensor([int])
        last_matching_index = (torch.tensor([last_src_ind], dtype = torch.int64), torch.tensor([last_tgt_ind], dtype = torch.int64))
        return indices_tensor, last_matching_index, event_mask_tensor
