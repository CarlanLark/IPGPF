# -*- coding: utf-8 -*-

from logging import raiseExceptions
from numpy import zeros
import torch
from torch import nn
import torch.nn.functional as F
import math
from collections import OrderedDict, namedtuple, defaultdict
from itertools import chain
import random
import copy

from . import transformer
from .ner_model import NERModel
from transformers import AutoTokenizer, AutoModel, BertTokenizer

from .set_criterion import SetCriterion
from .transformer import PointerNetwork
from .event_type import global_type2local_type
from .matcher import HungarianMatcher


DocSpanInfo = namedtuple(
    'DocSpanInfo', (
        'span_token_tup_list',  # [(span_token_id, ...), ...], num_spans
        'span_dranges_list',  # [[(sent_idx, char_s, char_e), ...], ...], num_spans
        'span_mention_range_list',  # [(mention_idx_s, mention_idx_e), ...], num_spans
        'mention_drange_list',  # [(sent_idx, char_s, char_e), ...], num_mentions
        'mention_type_list',  # [mention_type_id, ...], num_mentions
        'index2mention_type', # {mention_type_id: mention_type_string} such as 48: 'I-ReleasedDate'
        'role2id', #{role: id} such as 'Pledger': 18
        'gold_span_idx2pred_span_idx',
        'pred_event_arg_idxs_objs_list',
        'pred_event_type_idxs_list',
        'event_dag_info', 
    )
)


def get_doc_span_info_list(doc_token_types_list, doc_fea_list, use_gold_span=False):
    assert len(doc_token_types_list) == len(doc_fea_list)
    doc_span_info_list = []
    for doc_token_types, doc_fea in zip(doc_token_types_list, doc_fea_list):
        doc_token_type_mat = doc_token_types.tolist()  # [[token_type, ...], ...]

        # using extracted results is also ok
        # span_token_tup_list, span_dranges_list = extract_doc_valid_span_info(doc_token_type_mat, doc_fea)
        if use_gold_span:
            span_token_tup_list = doc_fea.span_token_ids_list
            span_dranges_list = doc_fea.span_dranges_list
        else:
            #BIO format
            span_token_tup_list, span_dranges_list = extract_doc_valid_span_info(doc_token_type_mat, doc_fea)

            if len(span_token_tup_list) == 0:
                # do not get valid entity span results,
                # just use gold spans to avoid crashing at earlier iterations
                # TODO: consider generate random negative spans
                span_token_tup_list = doc_fea.span_token_ids_list
                span_dranges_list = doc_fea.span_dranges_list

        # one span may have multiple mentions
        span_mention_range_list, mention_drange_list, mention_type_list = get_span_mention_info(
            span_dranges_list, doc_token_type_mat
        )

        #index to mention type dict
        index2mention_type = doc_fea.template_info['index2entity_label']
        role2id = doc_fea.template_info['role2id']

        gold_span_idx2pred_span_idx, pred_event_arg_idxs_objs_list, pred_event_type_idxs_list = doc_fea.generate_label(span_token_tup_list, return_miss=True)

        event_dag_info, _, missed_sent_idx_list = doc_fea.generate_dag_info_for(span_token_tup_list, return_miss=True)
        
        # doc_span_info will incorporate all span-level information needed for the event extraction
        doc_span_info = DocSpanInfo(
            span_token_tup_list, span_dranges_list, span_mention_range_list,
            mention_drange_list, mention_type_list, index2mention_type, role2id, 
            gold_span_idx2pred_span_idx, pred_event_arg_idxs_objs_list, pred_event_type_idxs_list, 
            event_dag_info, 
        )

        doc_span_info_list.append(doc_span_info)

    return doc_span_info_list

class NewEmbedding(nn.Module):
    def __init__(self, input_size, output_size, embedding):
        super(NewEmbedding, self).__init__()
        self.embedding = embedding
        self.MLP = MLP(input_size, output_size)
    
    def forward(self, x):
        return self.MLP(self.embedding(x))

class IterativeGenerator(nn.Module):
    def __init__(self, config, event_type_fields_pairs, ner_model=None):
        super(IterativeGenerator, self).__init__()
        # Note that for distributed training, you must ensure that
        # for any batch, all parameters need to be used

        self.config = config
        self.event_type_fields_pairs = event_type_fields_pairs

        if ner_model is None:
            self.ner_model = NERModel(config)
        else:
            self.ner_model = ner_model

        # all event tables
        self.event_tables = nn.ModuleList([
            EventTable(event_type, field_types, config.hidden_size)
            for event_type, field_types in self.event_type_fields_pairs
        ])

        # event type classifier: whether an event record will be reset to NoneType
        self.event_cls = nn.Linear(config.hidden_size, 2)

        # template selector: choose an event template for next generation
        # self.template_selector = nn.Linear(config.max_gen_num * config.hidden_size, config.hidden_size)

        #tokenizer
        if self.config.use_ernie:
            self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_model, do_lower_case = True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.bert_model, do_lower_case = True)

        # Embedding Layer
        if self.config.use_bert == False:
            self.embedding = self.ner_model.token_embedding.token_embedding
        else:
            if config.dim_reduce:
                self.embedding = NewEmbedding(config.bert_size, config.hidden_size, ner_model.bert.embeddings.word_embeddings)
            else:
                self.embedding = ner_model.bert.embeddings.word_embeddings
        if self.config.use_bert or self.config.use_ernie:
            self.role_embedding = nn.Embedding(100, config.hidden_size)


        if config.dim_reduce:
            self.MLP = MLP(config.bert_size, config.hidden_size)

        # sentence position indicator
        self.sent_pos_encoder = SentencePosEncoder(
            config.hidden_size, max_sent_num=config.max_sent_num, dropout=config.dropout
        )

        if self.config.use_token_role:
            self.ment_type_embedding = nn.Embedding(config.num_entity_labels, config.hidden_size)
            self.ment_type_layer_norm = transformer.LayerNorm(config.hidden_size)
            self.ment_type_dropout = nn.Dropout(config.dropout)

        # various attentive reducer
        if self.config.seq_reduce_type == 'AWA':
            self.doc_token_reducer = AttentiveReducer(config.hidden_size, dropout=config.dropout)
            self.span_token_reducer = AttentiveReducer(config.hidden_size, dropout=config.dropout)
            self.span_mention_reducer = AttentiveReducer(config.hidden_size, dropout=config.dropout)
        else:
            assert self.config.seq_reduce_type in {'MaxPooling', 'MeanPooling'}

        if self.config.use_doc_enc:
            # get doc-level context information for every mention and sentence
            self.doc_context_encoder = transformer.make_transformer_encoder(
                config.num_tf_layers, config.hidden_size, ff_size=config.ff_size, dropout=config.dropout
            )

        
        # Generator
        self.corpus_decoder = transformer.make_transformer_decoder(
            config.num_tf_layers, config.hidden_size, ff_size=config.ff_size, dropout=config.dropout
        )

        # Reference Decoder
            # get history event records as reference corpus
        self.generator = transformer.make_transformer_decoder(
            config.num_tf_layers, config.hidden_size, ff_size=config.ff_size, dropout=config.dropout
        )
        # pointer NN
        self.pointerNN = PointerNetwork(d_model = config.hidden_size)
        
        # matcher
        self.matcher = HungarianMatcher(config)
        
        # criterion
        self.criterion = SetCriterion(config, self.matcher)


        #template token type: role-0, context-1, pad-2
        if self.config.use_context_consistency:
            self.template_indicator_embed = nn.Embedding(3, config.hidden_size)
            # self.token_emb_reducer = nn.Linear(config.hidden_size, config.hidden_size // 8 * 7)
            # self.indicator_emb_reducer = nn.Linear(config.hidden_size, config.hidden_size // 8 * 1)

        if self.config.use_global_memory: 
            self.global_embedding = nn.Parameter(torch.randn(config.hidden_size))
            self.local_embedding = nn.Parameter(torch.randn(config.hidden_size))
        
        if self.config.memory_embedding_type == 'context':
            self.event_record_emb = nn.Parameter(torch.randn([2, config.hidden_size]))


    
    def convert_mention_label_index2role_id(self, mention_type_list, index2mention_type, role2id):
        ment_type_list = []
        role_id_list = []
        for index in mention_type_list:
            entity_label = index2mention_type[index]
            if len(entity_label) > 2 and entity_label[2:] in role2id:
                #'B-Pledger' from entity label to index of 'Pledger' from role
                role_id_list.append(role2id[entity_label[2:]])
                ment_type_list.append(entity_label[2:])
            else:
                role_id_list.append(role2id['Others']) #you can find details of role2id in template.py
                ment_type_list.append('Others')
        return role_id_list, ment_type_list
        
    def ment_type_encoding(self, doc_mention_emb, mention_type_list, index2mention_type, role2id):
        # mention_type_list is mention_type_index_list
        role_id_list, g_ment_type_list = self.convert_mention_label_index2role_id(mention_type_list, index2mention_type, role2id)
        role_id_tensor = torch.tensor(role_id_list, dtype=torch.long, device=doc_mention_emb.device, requires_grad=False)
        
        if self.config.use_bert or self.config.use_ernie:
            role_emb = self.role_embedding(role_id_tensor)
        else:
            if self.config.event_type_num == 13:
                mention_type_tensor = torch.tensor(mention_type_list, dtype=torch.long, device=doc_mention_emb.device, requires_grad=False)
                return self.ment_type_dropout(self.ment_type_layer_norm(doc_mention_emb + self.ment_type_embedding(mention_type_tensor))) 
            else:
                role_emb = self.embedding(role_id_tensor)
        if self.config.merge_entity:
            for idx, t in enumerate(g_ment_type_list):
                if t in global_type2local_type:
                    global_ment_type_tensor = torch.tensor([role2id[x] for x in global_type2local_type[t]], dtype=torch.long, device=doc_mention_emb.device, requires_grad=False)
                    role_emb[idx] = self.role_embedding(global_ment_type_tensor).mean(dim = 0) #[D]
        res = self.ment_type_dropout(self.ment_type_layer_norm(doc_mention_emb + role_emb)) 

        return res

    def get_doc_span_mention_emb(self, doc_token_emb, doc_span_info):
        if len(doc_span_info.mention_drange_list) == 0:
            doc_mention_emb = None
        else:
            # get mention context embeding
            # doc_mention_emb = torch.cat([
            #     # doc_token_emb[sent_idx, char_s:char_e, :].sum(dim=0, keepdim=True)
            #     doc_token_emb[sent_idx, char_s:char_e, :].max(dim=0, keepdim=True)[0]
            #     for sent_idx, char_s, char_e in doc_span_info.mention_drange_list
            # ])
            mention_emb_list = []
            for sent_idx, char_s, char_e in doc_span_info.mention_drange_list:
                mention_token_emb = doc_token_emb[sent_idx, char_s: char_e, :]  # [num_mention_tokens, hidden_size]
                if self.config.seq_reduce_type == 'AWA':
                    mention_emb = self.span_token_reducer(mention_token_emb)  # [hidden_size]
                elif self.config.seq_reduce_type == 'MaxPooling':
                    mention_emb = mention_token_emb.max(dim=0)[0]
                elif self.config.seq_reduce_type == 'MeanPooling':
                    mention_emb = mention_token_emb.mean(dim=0)
                else:
                    raise Exception('Unknown seq_reduce_type {}'.format(self.config.seq_reduce_type))
                mention_emb_list.append(mention_emb)
            doc_mention_emb = torch.stack(mention_emb_list, dim=0)

            # add sentence position embedding
            mention_sent_id_list = [drange[0] for drange in doc_span_info.mention_drange_list]
            doc_mention_emb = self.sent_pos_encoder(doc_mention_emb, sent_pos_ids=mention_sent_id_list)

            if self.config.use_token_role:
                # get mention type embedding
                mention_type_list, index2mention_type = doc_span_info.mention_type_list, doc_span_info.index2mention_type
                role2id = doc_span_info.role2id
                doc_mention_emb = self.ment_type_encoding(doc_mention_emb, mention_type_list, index2mention_type, role2id)

        return doc_mention_emb

    def get_batch_sent_emb(self, ner_token_emb, ner_token_masks, valid_sent_num_list):
        # From [ner_batch_size, sent_len, hidden_size] to [ner_batch_size, hidden_size]
        if self.config.seq_reduce_type == 'AWA':
            total_sent_emb = self.doc_token_reducer(ner_token_emb, masks=ner_token_masks)
        elif self.config.seq_reduce_type == 'MaxPooling':
            total_sent_emb = ner_token_emb.max(dim=1)[0]
        elif self.config.seq_reduce_type == 'MeanPooling':
            total_sent_emb = ner_token_emb.mean(dim=1)
        else:
            raise Exception('Unknown seq_reduce_type {}'.format(self.config.seq_reduce_type))

        total_sent_pos_ids = []
        for valid_sent_num in valid_sent_num_list:
            total_sent_pos_ids += list(range(valid_sent_num))
        total_sent_emb = self.sent_pos_encoder(total_sent_emb, sent_pos_ids=total_sent_pos_ids)

        return total_sent_emb
    
    def get_template_sent_emb(self, ner_token_emb, ner_token_masks = None, valid_sent_num_list = None):
        # From [ner_batch_size, sent_len, hidden_size] to [ner_batch_size, hidden_size]
        if self.config.seq_reduce_type == 'AWA':
            total_sent_emb = self.doc_token_reducer(ner_token_emb, masks=ner_token_masks)
        elif self.config.seq_reduce_type == 'MaxPooling':
            total_sent_emb = ner_token_emb.max(dim=1)[0]
        elif self.config.seq_reduce_type == 'MeanPooling':
            total_sent_emb = ner_token_emb.mean(dim=1)
        else:
            raise Exception('Unknown seq_reduce_type {}'.format(self.config.seq_reduce_type))

        total_sent_emb = self.sent_pos_encoder(total_sent_emb, sent_pos_ids=None)

        return total_sent_emb

    def get_reference_sent_emb(self, ner_token_emb, ner_token_masks = None, valid_sent_num_list = None):
        # From [ner_batch_size, sent_len, hidden_size] to [ner_batch_size, hidden_size]
        if self.config.seq_reduce_type == 'AWA':
            total_sent_emb = self.doc_token_reducer(ner_token_emb, masks=ner_token_masks)
        elif self.config.seq_reduce_type == 'MaxPooling':
            total_sent_emb = ner_token_emb.max(dim=1)[0]
        elif self.config.seq_reduce_type == 'MeanPooling':
            total_sent_emb = ner_token_emb.mean(dim=1)
        else:
            raise Exception('Unknown seq_reduce_type {}'.format(self.config.seq_reduce_type))

        total_sent_emb = self.sent_pos_encoder(total_sent_emb, sent_pos_ids=None)

        return total_sent_emb

    def get_doc_span_sent_context(self, doc_token_emb, doc_sent_emb, template_token_emb, doc_fea, doc_span_info, 
                                    re_span_rep = None, structure_mask=None):
        doc_mention_emb = self.get_doc_span_mention_emb(doc_token_emb, doc_span_info)

        # only consider actual sentences
        if doc_sent_emb.size(0) > doc_fea.valid_sent_num:
            doc_sent_emb = doc_sent_emb[:doc_fea.valid_sent_num, :]

        span_context_list = []

        if doc_mention_emb is None:
            if self.config.use_doc_enc:
                doc_sent_context = self.doc_context_encoder(doc_sent_emb.unsqueeze(0), None).squeeze(0)
                template_token_context = template_token_emb
            else:
                doc_sent_context = doc_sent_emb
        else:
            num_mentions = doc_mention_emb.size(0)
            num_sents = doc_sent_emb.size(0)

            if self.config.use_doc_enc:
                if self.config.use_role_span_interaction:
                    template_info = doc_fea.template_info
                    role2id = template_info['role2id'] #dict len = real roles + 1(Others)
                    num_roles = len(role2id) + 1
                    if self.config.use_bert or self.config.use_ernie:
                        role_emb = self.role_embedding(torch.arange(num_roles, device = doc_mention_emb.device, dtype = torch.long))#[real roles + 2, D]contain pad and Others-role-type, but we don't use them
                    else:
                        role_emb = self.embedding(torch.arange(num_roles, device = doc_mention_emb.device, dtype = torch.long))
                    # Size([1, num_mentions + num_valid_sents + num_roles, hidden_size])
                    total_ment_sent_emb = torch.cat([doc_mention_emb, doc_sent_emb, role_emb], dim=0).unsqueeze(0)
                else:
                    # Size([1, num_mentions + num_valid_sents, hidden_size])
                    total_ment_sent_emb = torch.cat([doc_mention_emb, doc_sent_emb], dim=0).unsqueeze(0)

                # size = [num_mentions+num_valid_sents, hidden_size]
                # here we do not need mask
                total_ment_sent_context = self.doc_context_encoder(total_ment_sent_emb, None).squeeze(0)

                # collect span context
                for mid_s, mid_e in doc_span_info.span_mention_range_list:
                    assert mid_e <= num_mentions
                    multi_ment_context = total_ment_sent_context[mid_s:mid_e]  # [num_mentions, hidden_size]

                    # span_context.size [1, hidden_size]
                    if self.config.seq_reduce_type == 'AWA':
                        span_context = self.span_mention_reducer(multi_ment_context, keepdim=True)
                    elif self.config.seq_reduce_type == 'MaxPooling':
                        span_context = multi_ment_context.max(dim=0, keepdim=True)[0]
                    elif self.config.seq_reduce_type == 'MeanPooling':
                        span_context = multi_ment_context.mean(dim=0, keepdim=True)
                    else:
                        raise Exception('Unknown seq_reduce_type {}'.format(self.config.seq_reduce_type))

                    span_context_list.append(span_context)

                # collect sent context
                doc_sent_context = total_ment_sent_context[num_mentions:num_mentions + num_sents, :]
                if self.config.use_role_span_interaction:
                    role_context = total_ment_sent_context[num_mentions+num_sents:num_mentions+num_sents + num_roles, :]
                    for event_idx, role_id_list in enumerate(template_info['event_type2role_id_list']):
                        role_id_tensor = torch.tensor(role_id_list, dtype=torch.long, requires_grad=False, device = doc_mention_emb.device)
                        argument_index_list = template_info['role_index_list'][event_idx]
                        argument_index_tensor = torch.tensor(argument_index_list, device = doc_mention_emb.device, dtype = torch.long)
                        template_token_emb[event_idx, argument_index_tensor, :] = role_context[role_id_tensor]
                template_token_context = template_token_emb
                

            else:
                # collect span context
                for mid_s, mid_e in doc_span_info.span_mention_range_list:
                    assert mid_e <= num_mentions
                    multi_ment_emb = doc_mention_emb[mid_s:mid_e]  # [num_mentions, hidden_size]

                    # span_context.size is [1, hidden_size]
                    if self.config.seq_reduce_type == 'AWA':
                        span_context = self.span_mention_reducer(multi_ment_emb, keepdim=True)
                    elif self.config.seq_reduce_type == 'MaxPooling':
                        span_context = multi_ment_emb.max(dim=0, keepdim=True)[0]
                    elif self.config.seq_reduce_type == 'MeanPooling':
                        span_context = multi_ment_emb.mean(dim=0, keepdim=True)
                    else:
                        raise Exception('Unknown seq_reduce_type {}'.format(self.config.seq_reduce_type))
                    span_context_list.append(span_context)

                # collect sent context
                doc_sent_context = doc_sent_emb
        

        return span_context_list, doc_sent_context, template_token_context

    def adjust_token_label(self, doc_token_labels_list):
        if self.config.use_token_role:  # do not use detailed token
            return doc_token_labels_list
        else:
            adj_doc_token_labels_list = []
            for doc_token_labels in doc_token_labels_list:
                entity_begin_mask = doc_token_labels % 2 == 1
                entity_inside_mask = (doc_token_labels != 0) & (doc_token_labels % 2 == 0)
                adj_doc_token_labels = doc_token_labels.masked_fill(entity_begin_mask, 1)
                adj_doc_token_labels = adj_doc_token_labels.masked_fill(entity_inside_mask, 2)

                adj_doc_token_labels_list.append(adj_doc_token_labels)
            return adj_doc_token_labels_list

    def get_local_context_info(self, doc_batch_dict, train_flag=False, use_gold_span=False):
        label_key = 'doc_token_labels'
        if train_flag or use_gold_span:
            assert label_key in doc_batch_dict
            need_label_flag = True
        else:
            need_label_flag = False

        if need_label_flag:
            doc_token_labels_list = self.adjust_token_label(doc_batch_dict[label_key])
        else:
            doc_token_labels_list = None

        batch_size = len(doc_batch_dict['ex_idx'])
        doc_token_ids_list = doc_batch_dict['doc_token_ids']
        doc_token_masks_list = doc_batch_dict['doc_token_masks']
        valid_sent_num_list = doc_batch_dict['valid_sent_num']


        # transform doc_batch into sent_batch
        ner_batch_idx_start_list = [0]
        ner_token_ids = []
        ner_token_masks = []
        ner_token_labels = [] if need_label_flag else None
        for batch_idx, valid_sent_num in enumerate(valid_sent_num_list):
            idx_start = ner_batch_idx_start_list[-1]
            idx_end = idx_start + valid_sent_num
            ner_batch_idx_start_list.append(idx_end)

            ner_token_ids.append(doc_token_ids_list[batch_idx])
            ner_token_masks.append(doc_token_masks_list[batch_idx])
            if need_label_flag:
                ner_token_labels.append(doc_token_labels_list[batch_idx])

        # [ner_batch_size, norm_sent_len]
        ner_token_ids = torch.cat(ner_token_ids, dim=0)
        ner_token_masks = torch.cat(ner_token_masks, dim=0)
        if need_label_flag:
            ner_token_labels = torch.cat(ner_token_labels, dim=0)

        # get ner output
        ner_token_emb, ner_loss, ner_token_preds = self.ner_model(
            ner_token_ids, ner_token_masks, label_ids=ner_token_labels,
            train_flag=train_flag, decode_flag=not use_gold_span,
        )
        if self.config.dim_reduce:
            ner_token_emb = self.MLP(ner_token_emb)

        if use_gold_span:  # definitely use gold span info
            ner_token_types = ner_token_labels
        else:
            ner_token_types = ner_token_preds


        # get sentence embedding
        ner_sent_emb = self.get_batch_sent_emb(ner_token_emb, ner_token_masks, valid_sent_num_list)

        assert sum(valid_sent_num_list) == ner_token_emb.size(0) == ner_sent_emb.size(0)

        # followings are all lists of tensors
        doc_token_emb_list = []
        doc_token_masks_list = []
        doc_token_types_list = []
        doc_sent_emb_list = []
        doc_sent_loss_list = []
        for batch_idx in range(batch_size):
            idx_start = ner_batch_idx_start_list[batch_idx]
            idx_end = ner_batch_idx_start_list[batch_idx+1]
            doc_token_emb_list.append(ner_token_emb[idx_start:idx_end, :, :])
            doc_token_masks_list.append(ner_token_masks[idx_start:idx_end, :])
            doc_token_types_list.append(ner_token_types[idx_start:idx_end, :])
            doc_sent_emb_list.append(ner_sent_emb[idx_start:idx_end, :])
            if ner_loss is not None:
                # every doc_sent_loss.size is [valid_sent_num]
                doc_sent_loss_list.append(ner_loss[idx_start:idx_end])

        return doc_token_emb_list, doc_token_masks_list, doc_token_types_list, doc_sent_emb_list, doc_sent_loss_list

    def get_event_cls_info(self, sent_context_emb, doc_fea, train_flag=True):
        doc_event_logps = []
        for event_idx, event_label in enumerate(doc_fea.event_type_labels):
            event_table = self.event_tables[event_idx]
            cur_event_logp = event_table(sent_context_emb=sent_context_emb)  # [1, hidden_size]
            doc_event_logps.append(cur_event_logp)
        doc_event_logps = torch.cat(doc_event_logps, dim=0)  # [num_event_types, 2]
        if train_flag:
            device = doc_event_logps.device
            doc_event_labels = torch.tensor(
                doc_fea.event_type_labels, device=device, dtype=torch.long, requires_grad=False
            )  # [num_event_types]
            doc_event_cls_loss = F.nll_loss(doc_event_logps, doc_event_labels, reduction='sum')
            doc_event_pred_list = doc_event_logps.argmax(dim=-1).tolist()
            return doc_event_cls_loss, doc_event_pred_list
        else:
            doc_event_pred_list = doc_event_logps.argmax(dim=-1).tolist()
            return doc_event_pred_list

    def get_single_pred(self, doc_sent_context, batch_span_context, template_token_context, doc_fea, doc_span_info, event_idx, train_flag = True, pre_filling_config = None, global_memory = None):
        #for single event generation
        # get device:
        device = batch_span_context.device
        #add None type role
        num_span = len(batch_span_context)
        role4None_embed = self.embedding(torch.tensor(self.tokenizer.convert_tokens_to_ids(['空']), device = device, dtype=torch.long))
        assert role4None_embed.size(0) == 1
        batch_span_context = torch.cat([batch_span_context, role4None_embed], dim = 0) #[num_span + 1, D]

        
        # load template info from doc feature
        template_index = event_idx # [1] template event index
        template_info = doc_fea.template_info
        template_token_ids_list = doc_fea.template_info['token_ids_list']
        # prepare template
        template_token_emb = template_token_context
        # template_sent_emb = self.get_template_sent_emb(template_token_emb)
        raw_template_token_emb = template_token_emb[template_index, :, :]
        # get memory
        if self.config.use_context_consistency:
            memory = torch.cat([batch_span_context, raw_template_token_emb, doc_sent_context], dim = 0) #[span_num + TL + doc_sent_num, D]
        else:
            memory = torch.cat([batch_span_context, doc_sent_context], dim = 0)

        if self.config.memory_embedding_type == 'text':
            # initial results tokens
            local_memory_token_list = [['*' for _ in range(self.config.max_refer_len)] for item in range(self.config.max_gen_num)]
            local_memory_token_ids_list = [self.tokenizer.convert_tokens_to_ids(x) for x in local_memory_token_list]
            # prepare local_memory
            local_memory_token_ids_tensor = torch.tensor(local_memory_token_ids_list, device = device, dtype = torch.long)    
            local_memory_token_emb = self.embedding(local_memory_token_ids_tensor) # [RN, RL, D]
            local_memory_sent_emb = self.get_reference_sent_emb(local_memory_token_emb) # [RN, D] 
            local_memory_sent_context = self.corpus_decoder(local_memory_sent_emb.unsqueeze(1), memory.unsqueeze(1)).squeeze(1) # [RN, D]
        elif self.config.memory_embedding_type == 'context':
            # initial local memory
            local_memory_sent_context = self.event_record_emb[1].unsqueeze(0) #[1, D]
            local_result_embedding = self.event_record_emb[1].unsqueeze(0) #[1, D]
        
        # initial results index
        local_memory_argument_score_list = []
        if self.config.use_context_consistency:
            local_memory_context_score_list = []
        if train_flag:
            loss = 0
            event_type_idxs_list = doc_span_info.pred_event_type_idxs_list[event_idx][:self.config.max_gen_num]
            event_arg_idxs_objs_list = doc_span_info.pred_event_arg_idxs_objs_list[event_idx][:self.config.max_gen_num]
            event_mask_tensor = torch.zeros(len(event_type_idxs_list), device = device)

        
        #build template input
        template_input = template_token_emb[template_index, :, :] #[TL, D]
        argument_index_list = template_info['role_index_list'][template_index]
        argument_index_tensor = torch.tensor(argument_index_list, device = device, dtype = torch.long)
        event_role_emb = template_token_emb[template_index, argument_index_tensor, :]

        num_roles = len(argument_index_tensor)

        #prepare template indicator embedding for template context consistency
        if self.config.use_context_consistency:
            #default to pad emb
            template_token_type_tensors = torch.tensor([2 for _ in range(len(template_input))], device = device, dtype = torch.long)
            #role 
            template_token_type_tensors[argument_index_list] = 0
            #context
            context_index_list = template_info['context_index_list'][template_index]
            context_index_tensor = torch.tensor(context_index_list, device = device, dtype = torch.long)
            template_token_type_tensors[context_index_list] = 1
            #embedding
            template_token_type_embed = self.template_indicator_embed(template_token_type_tensors) #[TL, D]
            template_input += template_token_type_embed

        
        #set mask matrix for arguments filling 
        argument_mask_tensor = torch.zeros([self.config.max_gen_num, num_roles], device = device, dtype = torch.int64)

        for iter_index in range(self.config.max_gen_num):
            #reset template input
            template_input[argument_index_tensor] = event_role_emb
            if self.config.use_context_consistency:
                template_input[argument_index_tensor] += template_token_type_embed[argument_index_tensor]

            #0. filling high prob args to template       
            if train_flag and pre_filling_config['use_pre_filling'] == 'ground_truth': # just use pre-filling in training
                if (event_mask_tensor == 0).int().sum().item() == 0 and iter_index > 0:
                    #select a history result
                    indices_tensor = indices_tensor.to(device)

                    prob_event = 1 + 1e-1 - local_memory_event_type_score.clone().detach()[indices_tensor[0]][:, 0]
                    pre_filling_iter_match = torch.distributions.Categorical(prob_event).sample()
                    pre_filling_iter = indices_tensor[0][pre_filling_iter_match]

                    #select pre-filling arguments
                    #get matched gold role
                    gt_iter = indices_tensor[1][pre_filling_iter_match]
                    gold_role = targets["role_label"]
                    gold_role_lists = [role_list for role_list in gold_role if role_list is not None]
                    gold_role = torch.tensor(gold_role_lists, device = device)
                    gold_role = gold_role[gt_iter] #[A]
                    
                    if self.config.ground_truth_pre_filling_type == 'decay':
                        gt_ratio = pre_filling_config['ground_truth_pre_filling_ratio'] * torch.ones_like(gold_role)
                        argument_mask_tensor[iter_index] = torch.distributions.Bernoulli(gt_ratio).sample().long()
                    elif self.config.ground_truth_pre_filling_type == 'threshold':
                        gt_prob = pred_role_score.clone().detach()[pre_filling_iter][:, gold_role].diag() #[A]
                        candidate = torch.where(gt_prob >= self.config.pre_filling_threshold, torch.ones_like(gt_prob), torch.zeros_like(gt_prob))
                        gt_ratio = pre_filling_config['ground_truth_pre_filling_ratio'] * candidate
                        argument_mask_tensor[iter_index] = torch.distributions.Bernoulli(gt_ratio).sample().long()

                    if (argument_mask_tensor[iter_index] == 1).int().sum().item() > 0:
                        mask_index = (argument_mask_tensor[iter_index] == 1).nonzero().view(-1)
                        span_index = gold_role[mask_index]
                        template_input[argument_index_tensor[mask_index]] = batch_span_context[span_index]
                        if self.config.use_context_consistency:
                            template_input[argument_index_tensor[mask_index]] += template_token_type_embed[argument_index_tensor[mask_index]]

            #1. predict cur iteration template filling
            # generation /local_result_embedding
            # template filling
            if global_memory != None:
                if self.config.memory_embedding_type == 'text':
                    gen = torch.cat([template_input.clone(), global_memory, local_memory_sent_context], dim = 0)#[TL+GN+LN, D]
                    gen_mask_list = template_info['key_padding_mask_list'][template_index] + [0 for _ in range(len(global_memory))] + [0 for _ in range(iter_index + 1)] + [1 for _ in range(len(local_memory_sent_context) - (iter_index + 1))]#[TL+GN+LN]
                elif self.config.memory_embedding_type == 'context':
                    gen = torch.cat([template_input.clone(), global_memory, local_memory_sent_context + local_result_embedding], dim = 0)
                    gen_mask_list = template_info['key_padding_mask_list'][template_index] + [0 for _ in range(len(global_memory) + len(local_memory_sent_context))]
                gen_mask_tensor = torch.tensor(gen_mask_list, device = device, dtype = torch.long).unsqueeze(0).byte().bool() #[1, TL+GN+LN]
            else:
                if self.config.memory_embedding_type == 'text':
                    gen = torch.cat([template_input.clone(), local_memory_sent_context], dim = 0)#[TL+RL, D]
                    gen_mask_list = template_info['key_padding_mask_list'][template_index] + [0 for _ in range(iter_index + 1)] + [1 for _ in range(len(local_memory_sent_context) - (iter_index + 1))]#[TL+RL]
                elif self.config.memory_embedding_type == 'context':
                    gen = torch.cat([template_input.clone(), local_memory_sent_context + local_result_embedding], dim = 0)#[TL+RL, D]
                    gen_mask_list = template_info['key_padding_mask_list'][template_index] + [0 for _ in range(len(local_memory_sent_context))]
                gen_mask_tensor = torch.tensor(gen_mask_list, device = device, dtype = torch.long).unsqueeze(0).byte().bool() #[1, TL+LN]
            template_filling = self.generator(gen.unsqueeze(1), memory = memory.unsqueeze(1), tgt_key_padding_mask = gen_mask_tensor).squeeze(1)[:len(template_input), :] #[TL, D]
            
            # select arguments
            argument_context = template_filling[argument_index_tensor] #[A, D]
            # pointer NN for select spans
            argument_score = self.pointerNN(tgt = argument_context, memory = batch_span_context).softmax(-1)#[A, 1, D] * [1, S, D] -> [A, S]
            argument_span_index = torch.max(argument_score, dim = -1)[1] #[A]
        
            # pointer NN for context consistency
            if self.config.use_context_consistency:
                context_context = template_filling[context_index_tensor] #[TCN, D] we denote number of template context as TCN
                context_score = self.pointerNN(tgt = context_context, memory = torch.cat([batch_span_context, raw_template_token_emb], dim = 0)).softmax(-1)#[TCN, 1, D] * [1, S+TL, D] -> [TCN, S+TL]
                local_memory_context_score_list.append(context_score.unsqueeze(0))
                      
            # append argument score 
            local_memory_argument_score_list.append(argument_score.unsqueeze(0))
            
            # update local memory
            if self.config.memory_embedding_type == 'text':
                # append filled template
                argument_span_index_list = argument_span_index.detach().cpu().numpy().tolist()
                unfilled_template_token_ids_list = copy.deepcopy(template_token_ids_list[template_index])
                unfilled_template_token_ids_list = [[x] for x in unfilled_template_token_ids_list]

                for index, argument_index in enumerate(argument_index_list):
                    argument_span_index = argument_span_index_list[index]
                    if argument_span_index == num_span:
                        argument_span_tup = self.tokenizer.convert_tokens_to_ids(['空'])
                    else:
                        argument_span_tup = doc_span_info.span_token_tup_list[argument_span_index]
                    argument_span_token_ids_list = list(argument_span_tup)
                    unfilled_template_token_ids_list[argument_index] = argument_span_token_ids_list# [id, id, ..., list(id, id, id), id, ...]
                filled_template_token_ids_list = list(chain.from_iterable(unfilled_template_token_ids_list))[:self.config.max_refer_len]
                
                local_memory_token_ids_list[iter_index][:len(filled_template_token_ids_list)] = filled_template_token_ids_list
                
                #2. update reference results
                local_memory_token_ids_tensor = torch.tensor(local_memory_token_ids_list, device = device, dtype = torch.long)
                local_memory_token_emb = self.embedding(local_memory_token_ids_tensor) # [RN, RL, D]
                local_memory_sent_emb = self.get_reference_sent_emb(local_memory_token_emb) # [RN, D]
                
                #result sentence mask
                local_memory_sent_mask_list = [0 for _ in range(iter_index + 1)] + [1 for _ in range(len(local_memory_sent_emb) - (iter_index + 1))] #[RN]
                local_memory_sent_mask_tensor = torch.tensor(local_memory_sent_mask_list, device = device, dtype = torch.long).unsqueeze(0).byte() #[1, RN]
                if global_memory != None:
                    global_memory_num = len(global_memory)
                    res_sent_emb = torch.cat([global_memory, local_memory_sent_emb], dim = 0)
                    res_sent_mask_list = [0 for _ in range(global_memory_num)] + local_memory_sent_mask_list
                    res_sent_mask_tensor = torch.tensor(res_sent_mask_list, device = device, dtype = torch.long).unsqueeze(0).byte() #[1, GN + RN]
                    res_sent_context = self.corpus_decoder(res_sent_emb.unsqueeze(1), memory.unsqueeze(1), tgt_key_padding_mask = res_sent_mask_tensor).squeeze(1) # [GN + RN, D]
                    global_sent_context, local_memory_sent_context = res_sent_context[:global_memory_num], res_sent_context[global_memory_num:]
                else:
                    local_memory_sent_context = self.corpus_decoder(local_memory_sent_emb.unsqueeze(1), memory.unsqueeze(1), tgt_key_padding_mask = local_memory_sent_mask_tensor).squeeze(1) # [RN, D]
                local_memory_event_type_score = self.event_cls(local_memory_sent_context).softmax(-1) # [RN, 1 + 1]
                local_memory_event_type = local_memory_event_type_score.max(dim = -1)[1] #[RN], 0 is event and 1 is not
                # update cls flag for local_memory results
                for index in range(iter_index + 1):
                    flag = local_memory_event_type[index].item()
                    N = len(template_info['draft_token_ids_list'][0])
                    local_memory_token_ids_list[index][0:N] = template_info['draft_token_ids_list'][flag]
            
            elif self.config.memory_embedding_type == 'context':
                res_token_emb = raw_template_token_emb.clone()
                res_token_emb[argument_index_tensor] = batch_span_context[argument_span_index]
                if iter_index == 0:
                    local_memory_token_emb = res_token_emb.unsqueeze(0) #[1, D]
                else:
                    local_memory_token_emb = torch.cat([local_memory_token_emb, res_token_emb.unsqueeze(0)], dim = 0)
                local_memory_sent_emb = self.get_reference_sent_emb(local_memory_token_emb) # [RN, D]
                # use global memory?
                if global_memory != None:
                    global_memory_num = len(global_memory)
                    res_sent_emb = torch.cat([global_memory, local_memory_sent_emb], dim = 0)
                    res_sent_context = self.corpus_decoder(res_sent_emb.unsqueeze(1), memory.unsqueeze(1)).squeeze(1) # [GN + RN, D]
                    global_sent_context, local_memory_sent_context = res_sent_context[:global_memory_num], res_sent_context[global_memory_num:]
                else:
                    local_memory_sent_context = self.corpus_decoder(local_memory_sent_emb.unsqueeze(1), memory.unsqueeze(1)).squeeze(1) # [RN, D]
                local_memory_event_type_score = self.event_cls(local_memory_sent_context).softmax(-1) # [RN, 1 + 1]
                local_memory_event_type = local_memory_event_type_score.max(dim = -1)[1] #[RN], 0 is event and 1 is not
                # update result embedding which means an event record is right or false
                local_result_embedding = self.event_record_emb[local_memory_event_type] #[RN, D]
            
            #3. compute loss
            pred_role_score = torch.cat(local_memory_argument_score_list, dim = 0) #[I, A, D] I = num_iters
            assert len(local_memory_event_type_score[:iter_index+1, :]) == len(pred_role_score)
            outputs = {'pred_doc_event': local_memory_event_type_score[:iter_index+1, :],'pred_role': pred_role_score}
            if self.config.use_context_consistency:
                pred_context_score = torch.cat(local_memory_context_score_list, dim = 0) #[I, TCN, D] I = num_iters
                outputs['pred_context'] = pred_context_score
            
            if train_flag:
                if len(event_type_idxs_list) != 0:
                    targets = {'doc_event_label': event_type_idxs_list, 'role_label': event_arg_idxs_objs_list}
                    
                    if self.config.use_context_consistency:
                        num_event = len(event_type_idxs_list)
                        template_context_idxs_list = [[(num_span + x) for x in context_index_list] for _ in range(num_event)] #[num_event, TCN]
                        targets['context_label'] = template_context_idxs_list
                    
                    iter_loss, event_mask_tensor, indices_tensor = self.criterion(outputs, targets, event_mask_tensor, argument_mask_tensor)
                    loss += iter_loss
                else:
                    loss += 0
        if global_memory != None:
            #Local: memory contains all generated events (both correct and uncorrect) while res only contains final selected events
            #Global: to prevent redundancy, global memory only contains final selected events from each event type.
            #Now we use global sentence embedding as global memory
            local_res_index = (local_memory_event_type == 0).nonzero().view(-1) #[LN] LN means local extracted events number
            local_res = local_memory_sent_emb[local_res_index] #[LN, D]
            global_memory = torch.cat([global_memory, local_res + self.global_embedding.unsqueeze(0)], dim = 0) #[GN + LN, D], G means global events number
        if train_flag:
            return loss, outputs, global_memory
        else:
            return outputs, global_memory
            
    def get_loss_on_doc(self, span_context_list, doc_sent_context, template_token_context, doc_fea, doc_span_info, pre_filling_config = None):
        if len(span_context_list) == 0:
            raise Exception('Error: doc_fea.ex_idx {} does not have valid span'.format(doc_fea.ex_idx))
        event_set_pred_loss = 0
        batch_span_context = torch.cat(span_context_list, dim=0)
        event_cls_loss, doc_event_pred_list = self.get_event_cls_info(doc_sent_context, doc_fea, train_flag=True)
        event_type_list = doc_fea.event_type_labels

        #initial global memory
        if self.config.use_global_memory:
            global_memory = self.global_embedding.unsqueeze(0) #[1, D]
        else:
            global_memory = None

        for event_idx, event_type in enumerate(event_type_list):
            if event_type != 0:
                set_pred_loss, _, global_memory = self.get_single_pred(doc_sent_context, batch_span_context, template_token_context, doc_fea, doc_span_info, event_idx, train_flag = True, pre_filling_config = pre_filling_config, global_memory = global_memory)
                event_set_pred_loss += set_pred_loss
        return event_set_pred_loss, event_cls_loss
    
    def get_mix_loss(self, doc_sent_loss_list, doc_event_loss_list, doc_span_info_list, re_loss = None):
        batch_size = len(doc_span_info_list)
        loss_batch_avg = 1.0 / batch_size
        lambda_1 = self.config.loss_lambda_1
        lambda_2 = self.config.loss_lambda_2
        lambda_3 = self.config.loss_lambda_3

        doc_ner_loss_list = []
        doc_event_type_loss_list = []
        doc_event_generate_loss_list = []
        for doc_sent_loss, doc_span_info in zip(doc_sent_loss_list, doc_span_info_list):
            # doc_sent_loss: Size([num_valid_sents])
            doc_ner_loss_list.append(doc_sent_loss.sum())

        for doc_event_loss in doc_event_loss_list:
            set_pred_loss, event_cls_loss = doc_event_loss
            doc_event_type_loss_list.append(event_cls_loss)
            doc_event_generate_loss_list.append(set_pred_loss)
            
        return loss_batch_avg * (lambda_1 * sum(doc_ner_loss_list) + lambda_2 * sum(doc_event_type_loss_list) + lambda_3 * sum(doc_event_generate_loss_list))

    def get_eval_on_doc(self, span_context_list, doc_sent_context, template_token_context, doc_fea, doc_span_info, pre_filling_config = None):
        if len(span_context_list) == 0:
            event_pred_list = []
            event_idx2obj_idx2field_idx2token_tup = []
            event_idx2event_decode_paths = []
            for event_idx in range(len(self.event_type_fields_pairs)):
                event_pred_list.append(0)
                event_idx2obj_idx2field_idx2token_tup.append(None)
                event_idx2event_decode_paths.append(None)

            return doc_fea.ex_idx, event_pred_list, event_idx2obj_idx2field_idx2token_tup, \
                doc_span_info, event_idx2event_decode_paths

        batch_span_context = torch.cat(span_context_list, dim=0)
        event_pred_list = self.get_event_cls_info(doc_sent_context, doc_fea, train_flag=False)
        num_entities = len(span_context_list)

        #initial global memory
        if self.config.use_global_memory:
            global_memory = self.global_embedding.unsqueeze(0) #[1, D]
        else:
            global_memory = None

        event_idx2obj_idx2field_idx2token_tup = []
        event_idx2event_decode_paths = []
        for event_idx, event_pred in enumerate(event_pred_list):
            if event_pred == 0:
                event_idx2obj_idx2field_idx2token_tup.append(None)
                event_idx2event_decode_paths.append(None)
            else:
                outputs, global_memory = self.get_single_pred(doc_sent_context, batch_span_context, template_token_context, doc_fea, doc_span_info, event_idx, train_flag=False, pre_filling_config = pre_filling_config, global_memory = global_memory)
                pred_event = outputs["pred_doc_event"].argmax(-1)  # [max_gen_num,event_types]
                pred_role = outputs["pred_role"].argmax(-1)  # [max_gen_num,num_roles,num_etities]
                obj_idx2field_idx2token_tup = self.pred2standard(pred_event, pred_role, doc_span_info, num_entities)
                if len(obj_idx2field_idx2token_tup) < 1:
                    event_idx2obj_idx2field_idx2token_tup.append(None)
                    event_idx2event_decode_paths.append(None)
                else:
                    event_idx2obj_idx2field_idx2token_tup.append(obj_idx2field_idx2token_tup)
                    event_idx2event_decode_paths.append(None)
        return doc_fea.ex_idx, event_pred_list, event_idx2obj_idx2field_idx2token_tup, doc_span_info, event_idx2event_decode_paths

    def pred2standard(self, pred_event_list, pred_role_list, doc_span_info, num_entities):
        obj_idx2field_idx2token_tup = []
        for pred_event, pred_role in zip(pred_event_list, pred_role_list):
            if int(pred_event) == 0:
                field_idx2token_tup = []
                for pred_role_index in pred_role:
                    if pred_role_index == num_entities:
                        field_idx2token_tup.append(None)
                    else:
                        field_idx2token_tup.append(doc_span_info.span_token_tup_list[pred_role_index])
                if field_idx2token_tup not in obj_idx2field_idx2token_tup:
                    obj_idx2field_idx2token_tup.append(field_idx2token_tup)
        return obj_idx2field_idx2token_tup

    def get_template_token_emb(self, doc_fea_list, device):
        template_token_emb_list = []
        for doc_fea in doc_fea_list:
            template_info = doc_fea.template_info
            template_token_ids_list = template_info['token_ids_list']
            # prepare template
            template_token_ids_tensor = torch.tensor(template_token_ids_list, device = device, dtype = torch.long)
            template_token_emb = self.embedding(template_token_ids_tensor) # [temp_num, max_temp_len, D]
            # template_sent_emb = self.get_template_sent_emb(template_token_emb)

            # insert query
            for event_idx in range(len(template_info['event_type2role_id_list'])):
                role_id_list = template_info['event_type2role_id_list'][event_idx]
                role_id_tensor = torch.tensor(role_id_list, dtype=torch.long, requires_grad=False).to(device)
                num_roles = len(role_id_list)
                if self.config.use_bert or self.config.use_ernie:
                    event_role_emb = self.role_embedding(role_id_tensor) #[A, D]
                else:
                    event_role_emb = self.embedding(role_id_tensor) #[A, D]
                #build template input
                argument_index_list = template_info['role_index_list'][event_idx]
                argument_index_tensor = torch.tensor(argument_index_list, device = device, dtype = torch.long)
                template_token_emb[event_idx, argument_index_tensor, :] = event_role_emb
            
            template_token_emb_list.append(template_token_emb)
        return template_token_emb_list

    def forward(self, doc_batch_dict, doc_features,
                train_flag=True, use_gold_span=False, teacher_prob=1,
                event_idx2entity_idx2field_idx=None, heuristic_type=None, 
                pre_filling_config = None):
        # Using scheduled sampling to gradually transit to predicted entity spans
        if train_flag and self.config.use_scheduled_sampling:
            # teacher_prob will gradually decrease outside
            if random.random() < teacher_prob:
                use_gold_span = True
            else:
                use_gold_span = False
        
        # get doc token-level local context
        doc_token_emb_list, doc_token_masks_list, doc_token_types_list, doc_sent_emb_list, doc_sent_loss_list = \
            self.get_local_context_info(
                doc_batch_dict, train_flag=train_flag, use_gold_span=use_gold_span,
            )

        # get doc feature objects
        ex_idx_list = doc_batch_dict['ex_idx']
        doc_fea_list = [doc_features[ex_idx] for ex_idx in ex_idx_list]
        
        # get template info list: the template num == doc num 
        # since we want different interactions between template emb and doc fea in each doc
        template_token_emb_list = self.get_template_token_emb(doc_fea_list, device = doc_token_emb_list[0].device)

        # get doc span-level info for event extraction
        doc_span_info_list = get_doc_span_info_list(doc_token_types_list, doc_fea_list, use_gold_span=use_gold_span)
        

        doc_span_context_list, doc_sent_context_list, template_token_context_list = [], [], []
        for batch_idx, ex_idx in enumerate(ex_idx_list):
            span_context_list, doc_sent_context, template_token_context = self.get_doc_span_sent_context(
            doc_token_emb_list[batch_idx], 
            doc_sent_emb_list[batch_idx], 
            template_token_emb_list[batch_idx], 
            doc_fea_list[batch_idx], 
            doc_span_info_list[batch_idx],
            )
            doc_span_context_list.append(span_context_list)
            doc_sent_context_list.append(doc_sent_context)
            template_token_context_list.append(template_token_context)

        if train_flag:
            doc_event_loss_list = []
            for batch_idx, ex_idx in enumerate(ex_idx_list):
                doc_event_loss_list.append(
                    self.get_loss_on_doc(
                        doc_span_context_list[batch_idx],
                        doc_sent_context_list[batch_idx],
                        template_token_context_list[batch_idx],
                        doc_fea_list[batch_idx],
                        doc_span_info_list[batch_idx],
                        pre_filling_config = pre_filling_config, 
                    )
                )

            mix_loss = self.get_mix_loss(doc_sent_loss_list, doc_event_loss_list, doc_span_info_list)

            return mix_loss
        else:
            # return a list object may not be supported by torch.nn.parallel.DataParallel
            # ensure to run it under the single-gpu mode
            eval_results = []
            for batch_idx, ex_idx in enumerate(ex_idx_list):
                eval_results.append(
                    self.get_eval_on_doc(
                        doc_span_context_list[batch_idx],
                        doc_sent_context_list[batch_idx],
                        template_token_context_list[batch_idx],
                        doc_fea_list[batch_idx],
                        doc_span_info_list[batch_idx],
                        pre_filling_config = pre_filling_config, 
                    )
                )
            return eval_results




def append_top_span_only(last_token_path_list, field_idx, field_idx2span_token_tup2dranges):
    new_token_path_list = []
    span_token_tup2dranges = field_idx2span_token_tup2dranges[field_idx]
    token_min_drange_list = [
        (token_tup, dranges[0]) for token_tup, dranges in span_token_tup2dranges.items()
    ]
    token_min_drange_list.sort(key=lambda x: x[1])

    for last_token_path in last_token_path_list:
        new_token_path = list(last_token_path)
        if len(token_min_drange_list) == 0:
            new_token_path.append(None)
        else:
            token_tup = token_min_drange_list[0][0]
            new_token_path.append(token_tup)

        new_token_path_list.append(new_token_path)

    return new_token_path_list


def append_all_spans(last_token_path_list, field_idx, field_idx2span_token_tup2dranges):
    new_token_path_list = []
    span_token_tup2dranges = field_idx2span_token_tup2dranges[field_idx]

    for last_token_path in last_token_path_list:
        for token_tup in span_token_tup2dranges.keys():
            new_token_path = list(last_token_path)
            new_token_path.append(token_tup)
            new_token_path_list.append(new_token_path)

        if len(span_token_tup2dranges) == 0:  # ensure every last path will be extended
            new_token_path = list(last_token_path)
            new_token_path.append(None)
            new_token_path_list.append(new_token_path)

    return new_token_path_list


class AttentiveReducer(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(AttentiveReducer, self).__init__()

        self.hidden_size = hidden_size
        self.att_norm = math.sqrt(self.hidden_size)

        self.fc = nn.Linear(hidden_size, 1, bias=False)
        self.att = None

        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_token_emb, masks=None, keepdim=False):
        # batch_token_emb: Size([*, seq_len, hidden_size])
        # masks: Size([*, seq_len]), 1: normal, 0: pad

        query = self.fc.weight
        if masks is None:
            att_mask = None
        else:
            att_mask = masks.unsqueeze(-2)  # [*, 1, seq_len]

        # batch_att_emb: Size([*, 1, hidden_size])
        # self.att: Size([*, 1, seq_len])
        batch_att_emb, self.att = transformer.attention(
            query, batch_token_emb, batch_token_emb, mask=att_mask
        )

        batch_att_emb = self.dropout(self.layer_norm(batch_att_emb))

        if keepdim:
            return batch_att_emb
        else:
            return batch_att_emb.squeeze(-2)

    def extra_repr(self):
        return 'hidden_size={}, att_norm={}'.format(self.hidden_size, self.att_norm)


class SentencePosEncoder(nn.Module):
    def __init__(self, hidden_size, max_sent_num=100, dropout=0.1):
        super(SentencePosEncoder, self).__init__()

        self.embedding = nn.Embedding(max_sent_num, hidden_size)
        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_elem_emb, sent_pos_ids=None):
        if sent_pos_ids is None:
            num_elem = batch_elem_emb.size(-2)
            sent_pos_ids = torch.arange(
                num_elem, dtype=torch.long, device=batch_elem_emb.device, requires_grad=False
            )
        elif not isinstance(sent_pos_ids, torch.Tensor):
            sent_pos_ids = torch.tensor(
                sent_pos_ids, dtype=torch.long, device=batch_elem_emb.device, requires_grad=False
            )

        batch_pos_emb = self.embedding(sent_pos_ids)
        out = batch_elem_emb + batch_pos_emb
        out = self.dropout(self.layer_norm(out))

        return out


class MentionTypeEncoder(nn.Module):
    def __init__(self, hidden_size, num_ment_types, dropout=0.1):
        super(MentionTypeEncoder, self).__init__()

        self.embedding = nn.Embedding(num_ment_types, hidden_size)
        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_mention_emb, mention_type_ids):
        if not isinstance(mention_type_ids, torch.Tensor):
            mention_type_ids = torch.tensor(
                mention_type_ids, dtype=torch.long, device=batch_mention_emb.device, requires_grad=False
            )

        batch_mention_type_emb = self.embedding(mention_type_ids)
        out = batch_mention_emb + batch_mention_type_emb
        out = self.dropout(self.layer_norm(out))

        return out


class EventTable(nn.Module):
    def __init__(self, event_type, field_types, hidden_size):
        super(EventTable, self).__init__()

        self.event_type = event_type
        self.field_types = field_types
        self.num_fields = len(field_types)
        self.hidden_size = hidden_size

        self.event_cls = nn.Linear(hidden_size, 2)  # 0: NA, 1: trigger this event
        # self.field_cls_list = nn.ModuleList(
        #     # 0: NA, 1: trigger this field
        #     [nn.Linear(hidden_size, 2) for _ in range(self.num_fields)]
        # )

        # used to aggregate sentence and span embedding
        self.event_query = nn.Parameter(torch.Tensor(1, self.hidden_size))
        # used for fields that do not contain any valid span
        # self.none_span_emb = nn.Parameter(torch.Tensor(1, self.hidden_size))
        # used for aggregating history filled span info
        # self.field_queries = nn.ParameterList(
        #     [nn.Parameter(torch.Tensor(1, self.hidden_size)) for _ in range(self.num_fields)]
        # )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_size)
        self.event_query.data.uniform_(-stdv, stdv)
        # self.none_span_emb.data.uniform_(-stdv, stdv)
        # for fq in self.field_queries:
        #     fq.data.uniform_(-stdv, stdv)

    def forward(self, sent_context_emb=None, batch_span_emb=None, field_idx=None):
        assert (sent_context_emb is None) ^ (batch_span_emb is None)

        if sent_context_emb is not None:  # [num_spans+num_sents, hidden_size]
            # doc_emb.size = [1, hidden_size]
            doc_emb, _ = transformer.attention(self.event_query, sent_context_emb, sent_context_emb)
            doc_pred_logits = self.event_cls(doc_emb)
            doc_pred_logp = F.log_softmax(doc_pred_logits, dim=-1)

            return doc_pred_logp

        # if batch_span_emb is not None:
        #     assert field_idx is not None
        #     # span_context_emb: [batch_size, hidden_size] or [hidden_size]
        #     if batch_span_emb.dim() == 1:
        #         batch_span_emb = batch_span_emb.unsqueeze(0)
        #     span_pred_logits = self.field_cls_list[field_idx](batch_span_emb)
        #     span_pred_logp = F.log_softmax(span_pred_logits, dim=-1)

        #     return span_pred_logp

    def extra_repr(self):
        return 'event_type={}, num_fields={}, hidden_size={}'.format(
            self.event_type, self.num_fields, self.hidden_size
        )


class MLP(nn.Module):
    """Implements Multi-layer Perception."""

    def __init__(self, input_size, output_size, mid_size=None, num_mid_layer=1, dropout=0.1):
        super(MLP, self).__init__()

        assert num_mid_layer >= 1
        if mid_size is None:
            mid_size = input_size

        self.input_fc = nn.Linear(input_size, mid_size)
        self.out_fc = nn.Linear(mid_size, output_size)
        if num_mid_layer > 1:
            self.mid_fcs = nn.ModuleList(
                nn.Linear(mid_size, mid_size) for _ in range(num_mid_layer-1)
            )
        else:
            self.mid_fcs = []
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.input_fc(x)))
        for mid_fc in self.mid_fcs:
            x = self.dropout(F.relu(mid_fc(x)))
        x = self.out_fc(x)
        return x


def get_span_mention_info(span_dranges_list, doc_token_type_list):
    span_mention_range_list = []
    mention_drange_list = []
    mention_type_list = []
    for span_dranges in span_dranges_list:
        ment_idx_s = len(mention_drange_list)
        for drange in span_dranges:
            mention_drange_list.append(drange)
            sent_idx, char_s, char_e = drange
            mention_type_list.append(doc_token_type_list[sent_idx][char_s])
        ment_idx_e = len(mention_drange_list)
        span_mention_range_list.append((ment_idx_s, ment_idx_e))

    return span_mention_range_list, mention_drange_list, mention_type_list


def extract_doc_valid_span_info(doc_token_type_mat, doc_fea):
    doc_token_id_mat = doc_fea.doc_token_ids.tolist()
    doc_token_mask_mat = doc_fea.doc_token_masks.tolist()

    # [(token_id_tuple, (sent_idx, char_s, char_e)), ...]
    span_token_drange_list = []

    valid_sent_num = doc_fea.valid_sent_num
    for sent_idx in range(valid_sent_num):
        seq_token_id_list = doc_token_id_mat[sent_idx]
        seq_token_mask_list = doc_token_mask_mat[sent_idx]
        seq_token_type_list = doc_token_type_mat[sent_idx]
        seq_len = len(seq_token_id_list)

        char_s = 0
        while char_s < seq_len:
            if seq_token_mask_list[char_s] == 0:
                break

            entity_idx = seq_token_type_list[char_s]

            if entity_idx % 2 == 1:
                char_e = char_s + 1
                while char_e < seq_len and seq_token_mask_list[char_e] == 1 and \
                        seq_token_type_list[char_e] == entity_idx + 1:
                    char_e += 1

                token_tup = tuple(seq_token_id_list[char_s:char_e])
                drange = (sent_idx, char_s, char_e)

                span_token_drange_list.append((token_tup, drange))

                char_s = char_e
            else:
                char_s += 1

    span_token_drange_list.sort(key=lambda x: x[-1])  # sorted by drange = (sent_idx, char_s, char_e)
    # drange is exclusive and sorted
    token_tup2dranges = OrderedDict()
    for token_tup, drange in span_token_drange_list:
        if token_tup not in token_tup2dranges:
            token_tup2dranges[token_tup] = []
        token_tup2dranges[token_tup].append(drange)

    span_token_tup_list = list(token_tup2dranges.keys())
    span_dranges_list = list(token_tup2dranges.values())

    return span_token_tup_list, span_dranges_list

def extract_doc_valid_span_info_for_bieos(doc_token_type_mat, doc_fea):
    doc_token_id_mat = doc_fea.doc_token_ids.tolist()
    doc_token_mask_mat = doc_fea.doc_token_masks.tolist()

    # [(token_id_tuple, (sent_idx, char_s, char_e)), ...]
    span_token_drange_list = []

    valid_sent_num = doc_fea.valid_sent_num
    for sent_idx in range(valid_sent_num):
        seq_token_id_list = doc_token_id_mat[sent_idx]
        seq_token_mask_list = doc_token_mask_mat[sent_idx]
        seq_token_type_list = doc_token_type_mat[sent_idx]
        seq_len = len(seq_token_id_list)

        char_s = 0
        while char_s < seq_len:
            if seq_token_mask_list[char_s] == 0:
                break

            entity_idx = seq_token_type_list[char_s]
            # Get the valid token span. the division is 2 since the encoding format is BIO, otherwise 4 for BIEOS.
            if entity_idx % 4 == 1:
                char_e = char_s + 1
                while char_e < seq_len and seq_token_mask_list[char_e] == 1 and \
                        seq_token_type_list[char_e] == entity_idx + 1:
                    char_e += 1
                if char_e == seq_len:
                    break
                elif char_e >= char_s + 1 and seq_token_type_list[char_e] == entity_idx + 2:
                    token_tup = tuple(seq_token_id_list[char_s: char_e + 1])
                    drange = (sent_idx, char_s, char_e + 1)
                    char_e += 1
                    span_token_drange_list.append((token_tup, drange))
                char_s = char_e
                #     char_s = char_e
                # else:
                #     char_s = char_e

            elif entity_idx % 4 == 0 and entity_idx != 0:
                char_e = char_s + 1
                token_tup = tuple(seq_token_id_list[char_s: char_e])
                drange = (sent_idx, char_s, char_e)
                span_token_drange_list.append((token_tup, drange))
                char_s = char_e
            else:
                char_s += 1

    span_token_drange_list.sort(key=lambda x: x[-1])  # sorted by drange = (sent_idx, char_s, char_e)
    # drange is exclusive and sorted
    token_tup2dranges = OrderedDict()
    for token_tup, drange in span_token_drange_list:
        if token_tup not in token_tup2dranges:
            token_tup2dranges[token_tup] = []
        token_tup2dranges[token_tup].append(drange)

    span_token_tup_list = list(token_tup2dranges.keys())
    span_dranges_list = list(token_tup2dranges.values())

    return span_token_tup_list, span_dranges_list


def get_batch_span_label(num_spans, cur_span_idx_set, device):
    # prepare span labels for this field and this path
    span_field_labels = [
        1 if span_idx in cur_span_idx_set else 0 for span_idx in range(num_spans)
    ]

    batch_field_label = torch.tensor(
        span_field_labels, dtype=torch.long, device=device, requires_grad=False
    )  # [num_spans], val \in {0, 1}

    return batch_field_label


def get_one_key_sent_event(key_sent_idx, num_fields, field_idx2span_token_tup2dranges):
    field_idx2token_tup = []
    for field_idx in range(num_fields):
        token_tup2dranges = field_idx2span_token_tup2dranges[field_idx]

        # find the closest token_tup to the key sentence
        best_token_tup = None
        best_dist = 10000
        for token_tup, dranges in token_tup2dranges.items():
            for sent_idx, _, _ in dranges:
                cur_dist = abs(sent_idx - key_sent_idx)
                if cur_dist < best_dist:
                    best_token_tup = token_tup
                    best_dist = cur_dist

        field_idx2token_tup.append(best_token_tup)
    return field_idx2token_tup


def get_many_key_sent_event(key_sent_idx, num_fields, field_idx2span_token_tup2dranges):
    # get key_field_idx contained in key event sentence
    key_field_idx2token_tup_set = defaultdict(lambda: set())
    for field_idx, token_tup2dranges in field_idx2span_token_tup2dranges.items():
        assert field_idx < num_fields
        for token_tup, dranges in token_tup2dranges.items():
            for sent_idx, _, _ in dranges:
                if sent_idx == key_sent_idx:
                    key_field_idx2token_tup_set[field_idx].add(token_tup)

    field_idx2token_tup_list = []
    while len(key_field_idx2token_tup_set) > 0:
        # get key token tup candidates according to the distance in the sentence
        prev_field_idx = None
        prev_token_cand = None
        key_field_idx2token_cand = {}
        for key_field_idx, token_tup_set in key_field_idx2token_tup_set.items():
            assert len(token_tup_set) > 0

            if prev_token_cand is None:
                best_token_tup = token_tup_set.pop()
            else:
                prev_char_range = field_idx2span_token_tup2dranges[prev_field_idx][prev_token_cand][0][1:]
                best_dist = 10000
                best_token_tup = None
                for token_tup in token_tup_set:
                    cur_char_range = field_idx2span_token_tup2dranges[key_field_idx][token_tup][0][1:]
                    cur_dist = min(
                        abs(cur_char_range[1] - prev_char_range[0]),
                        abs(cur_char_range[0] - prev_char_range[1])
                    )
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_token_tup = token_tup
                token_tup_set.remove(best_token_tup)

            key_field_idx2token_cand[key_field_idx] = best_token_tup
            prev_field_idx = key_field_idx
            prev_token_cand = best_token_tup

        field_idx2token_tup = []
        for field_idx in range(num_fields):
            token_tup2dranges = field_idx2span_token_tup2dranges[field_idx]

            if field_idx in key_field_idx2token_tup_set:
                token_tup_set = key_field_idx2token_tup_set[field_idx]
                if len(token_tup_set) == 0:
                    del key_field_idx2token_tup_set[field_idx]
                token_tup = key_field_idx2token_cand[field_idx]
                field_idx2token_tup.append(token_tup)
            else:
                # find the closest token_tup to the key sentence
                best_token_tup = None
                best_dist = 10000
                for token_tup, dranges in token_tup2dranges.items():
                    for sent_idx, _, _ in dranges:
                        cur_dist = abs(sent_idx - key_sent_idx)
                        if cur_dist < best_dist:
                            best_token_tup = token_tup
                            best_dist = cur_dist

                field_idx2token_tup.append(best_token_tup)

        field_idx2token_tup_list.append(field_idx2token_tup)

    return field_idx2token_tup_list


