"""
The codes are based on https://github.com/HangYang-NLP/DE-PPN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F




class SetCriterion(nn.Module):
    def __init__(self, config, matcher, na_coef = 0.1, losses = ["event", "role"]):
        super().__init__()
        self.config = config
        self.matcher = matcher
        self.num_classes = config.event_type_classes
        self.losses = losses
        if config.event_type_weight:
            self.type_weight = torch.tensor(config.event_type_weight).cuda()
        else:
            self.type_weight = torch.ones(self.num_classes).cuda()
            self.type_weight[-1] = na_coef
        self.register_buffer('rel_weight', self.type_weight)
        

    def forward(self, outputs, targets, event_mask_tensor, argument_mask_tensor):
        indices_tensor, last_matching_index, event_mask_tensor = self.matcher(outputs, targets, event_mask_tensor)
        losses = self.get_role_loss(outputs, targets, indices_tensor, last_matching_index, argument_mask_tensor)
        if self.config.use_context_consistency:
            losses += self.get_context_loss(outputs, targets, last_matching_index)
        return losses, event_mask_tensor, indices_tensor

    def get_context_loss(self, outputs, targets, last_matching_index):
        pred_context = outputs['pred_context'] 
        gold_context = targets['context_label']
        gold_context_tensor = torch.tensor(gold_context).cuda()

        selected_pred_context_tensor = pred_context[last_matching_index[0]]
        selected_gold_context_tensor = gold_context_tensor[last_matching_index[1]].clone()

        context_loss = F.cross_entropy(selected_pred_context_tensor.flatten(0, 1), selected_gold_context_tensor.flatten(0, 1))
        return context_loss



    def get_role_loss(self, outputs, targets, indices_tensor, last_matching_index, argument_mask_tensor = None):
        num_sets, num_roles, num_entities = outputs["pred_role"].size()

        pred_event = outputs["pred_doc_event"]
        gold_event = targets["doc_event_label"]
        gold_event_tensor = torch.tensor(gold_event).cuda()

        pred_role = outputs["pred_role"] 
        gold_role = targets["role_label"]
        gold_role_tensor = torch.tensor(gold_role).cuda()
        if self.num_classes == 2:
            gold_event_tensor = torch.zeros(gold_event_tensor.size()).long().cuda()

        selected_pred_role_tensor = pred_role[last_matching_index[0]]
        selected_gold_role_tensor = gold_role_tensor[last_matching_index[1]].clone()
        #select mask tensor and mask label
        ignore_index = -1
        if argument_mask_tensor != None:
            selected_mask_role_tensor = argument_mask_tensor[last_matching_index[0]]
            selected_gold_role_tensor = torch.where(selected_mask_role_tensor == 1, ignore_index, selected_gold_role_tensor)

        gold_event_label = torch.full(pred_event.shape[:1], self.num_classes -1, dtype=torch.int64).cuda()
        assert gold_event_label[indices_tensor[0]].size() == gold_event_tensor[indices_tensor[1]].size()
        gold_event_label[indices_tensor[0]] = gold_event_tensor[indices_tensor[1]]
        event_type_loss = F.cross_entropy(pred_event, gold_event_label, weight=self.type_weight)

        role_weight = torch.ones(num_entities).cuda()
        role_weight[-1] = self.config.neg_field_loss_scaling
        #if all arguments have been filled, we set loss=0 and skip this training iter
        if (selected_gold_role_tensor == ignore_index).int().sum().item() == selected_gold_role_tensor.numel():
            role_loss, event_type_loss = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        else:
            role_loss = F.cross_entropy(selected_pred_role_tensor.flatten(0, 1), selected_gold_role_tensor.flatten(0, 1), weight=role_weight, ignore_index = ignore_index)
       
        losses = event_type_loss + role_loss

        return losses

    def get_role_loss_only(self, outputs, targets, indices_tensor, argument_mask_tensor = None):
        num_sets, num_roles, num_entities = outputs["pred_role"].size()

        pred_role = outputs["pred_role"]
        gold_role = targets["role_label"]
        gold_role_tensor = torch.tensor(gold_role).cuda()

        selected_pred_role_tensor = pred_role[indices_tensor[0]]
        selected_gold_role_tensor = gold_role_tensor[indices_tensor[1]].clone()

        ignore_index = -1
        if argument_mask_tensor != None:
            selected_mask_role_tensor = argument_mask_tensor[indices_tensor[0]]
            selected_gold_role_tensor = torch.where(selected_mask_role_tensor == 1, ignore_index, selected_gold_role_tensor)

        key_role_weight = torch.ones(num_entities).cuda()
        key_role_weight[-1] = 1
        key_role_loss = F.cross_entropy(selected_pred_role_tensor[:,:-2].flatten(0, 1), selected_gold_role_tensor[:, :-2].flatten(0, 1), weight=key_role_weight, ignore_index = ignore_index)

        rest_role_weight = torch.ones(num_entities).cuda()
        rest_role_weight[-1] = 0.2
        rest_role_loss = F.cross_entropy(selected_pred_role_tensor[:,-2:].flatten(0, 1), selected_gold_role_tensor[:, -2:].flatten(0, 1), weight=rest_role_weight, ignore_index = ignore_index)

        role_loss = key_role_loss + rest_role_loss

        losses = role_loss 

        return losses


