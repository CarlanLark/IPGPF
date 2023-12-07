from .utils import bert_chinese_char_tokenize
from .event_type import common_fields

class Template(object):
    def __init__(self, tokenizer, event_type_fields_list, template_text_type):
        #note that event_type_fields_pairs = list(event_type_fields_list) == event_type_fields_list
        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.vocab
        self.event_type_fields_list = event_type_fields_list
        #set template
        self.template_tokens, self.template_roles = self.set_template(text_type = template_text_type)
        self.template_draft_token_ids_list = [self.tokenizer.convert_tokens_to_ids(['正', '确', '事', '件']), self.tokenizer.convert_tokens_to_ids(['参', '考', '记', '录'])]
        #tamplate
        self.template_token_ids_list, self.template_key_padding_mask_list = self.get_template_token_ids_tensor(self.template_tokens, self.event_type_fields_list, self.tokenizer)
        #tamplate role
        self.role_type2template_role_index_dict = self.get_role_type2template_role_index_dict(self.template_tokens, self.template_roles, self.tokenizer)
        self.event_type2template_role_index_dict = self.get_event_type2template_role_index_dict(self.role_type2template_role_index_dict, self.event_type_fields_list)
        self.event_type2template_role_index_list = self.get_event_type2template_role_index_list(self.role_type2template_role_index_dict, self.event_type_fields_list)
        #tamplate context : for context consistency
        self.event_type2template_context_index_dict = self.get_event_type2template_context_index_dict(self.template_tokens, self.tokenizer)
        self.event_type2template_context_index_list = self.get_event_type2template_context_index_list(self.event_type2template_context_index_dict, self.event_type_fields_list)
        #role embedding info
        self.event_type2role_id_list, self.role2id = self.get_event_type2role_id_list(self.tokenizer, self.event_type_fields_list)
        #tamplate_info
        self.template_info = {
            'token_ids_list': self.template_token_ids_list, 
            'key_padding_mask_list': self.template_key_padding_mask_list, 
            'draft_token_ids_list': self.template_draft_token_ids_list, 
            # role_index above means positional index in template sentence
            'role_index_list': self.event_type2template_role_index_list, 
            'context_index_list': self.event_type2template_context_index_list, 
            # role_id above means the id of role-label-embedding such as '[Pledger]'
            'event_type2role_id_list': self.event_type2role_id_list, 
            'role2id': self.role2id, #dict
        }

    def set_template(self, text_type):
        if text_type == 'DuEE-fin-1':
            template_tokens = \
                {
                '质押': '正确事件：#########。', 
                '股份回购': '正确事件：#######。', 
                '解除质押': '正确事件：#########。', 
                '被约谈': '正确事件：####。', 
                '企业收购': '正确事件：######。', 
                '股东增持': '正确事件：#########。', 
                '高管变动': '正确事件：########。', 
                '中标': '正确事件：######。', 
                '公司上市': '正确事件：########。', 
                '企业融资': '正确事件：#######。', 
                '亏损': '正确事件：#####。', 
                '股东减持': '正确事件：#########。', 
                '企业破产': '正确事件：#####。', 
                'NoneType': '参考记录：本记录为事件抽取的草稿参考，并不作为事件抽取的正式结果。'
                }
            template_roles = \
                {
                '质押': ["质押股票/股份数量","质押物占持股比","质押物占总股比","事件时间","质押方","质押物","披露时间","质押物所属公司","质权方",], 
                '股份回购': ["回购股份数量", "回购完成时间", "交易金额", "每股交易价格", "回购方", "占公司总股本比例", "披露时间"], 
                '解除质押': ["质权方","质押物占总股比","质押方","事件时间","质押股票/股份数量","质押物所属公司","质押物","质押物占持股比","披露时间",], 
                '被约谈': ["约谈机构", "被约谈时间", "披露时间", "公司名称"], 
                '企业收购': ["被收购方", "收购标的", "交易金额", "收购方", "收购完成时间", "披露时间"], 
                '股东增持': ["每股交易价格","交易金额","增持部分占所持比例","交易完成时间","增持方","交易股票/股份数量","增持部分占总股本比例","股票简称","披露时间",], 
                '高管变动': ["变动后职位", "任职公司", "高管姓名", "披露日期", "变动类型", "事件时间", "高管职位", "变动后公司名称"], 
                '中标': ["中标金额", "披露日期", "招标方", "中标日期", "中标标的", "中标公司"], 
                '公司上市': ["募资金额", "事件时间", "证券代码", "环节", "发行价格", "上市公司", "披露时间", "市值"], 
                '企业融资': ["融资金额", "事件时间", "被投资方", "领投方", "融资轮次", "披露时间", "投资方"], 
                '亏损': ["亏损变化", "财报周期", "净亏损", "披露时间", "公司名称"], 
                '股东减持': ["减持方","每股交易价格","交易金额","减持部分占所持比例","交易完成时间","交易股票/股份数量","减持部分占总股本比例","股票简称","披露时间",], 
                '企业破产': ["债务规模", "破产公司", "债权人", "破产时间", "披露时间"], 
                'NoneType': [], 
                }
        elif text_type == 'ChFinAnn-1':
            template_tokens = \
                {
                'EquityOverweight': '正确事件：股票增持：#以#/股的平均价格购入#股份，购入时间从#开始，至#截止，购入后，仍持有公司股份#。', 
                'EquityUnderweight': '正确事件：股票减持：#以#/股的平均价格售出#股份，售出时间从#开始，至#截止，售出后，仍持有公司股份#。', 
                'EquityFreeze': '正确事件：股权冻结：在#，#冻结或解除冻结了#所持有的股票#，冻结时间从#开始，到#截止，目前仍持有公司股份#，占公司总股本的#。',  
                'EquityPledge': '正确事件：股权质押：在#，#质押给或解除质押#共#股份，质押时间从#开始，到#截止，目前仍持有公司股份#，占公司总股本的#，累积质押#股。', 
                'EquityRepurchase': '正确事件：股权回购：#以#/股的最高价和#/股的最低价回购了#股份，回购时间为#，支付总金额约为#。', 
                'NoneType': '参考记录：本记录为事件抽取的草稿参考，并不作为事件抽取的正式结果。'
                }
            template_roles = \
                {
                'EquityOverweight': ['EquityHolder', 'AveragePrice', 'TradedShares', 'StartDate', 'EndDate', 'LaterHoldingShares'], 
                'EquityUnderweight': ['EquityHolder', 'AveragePrice', 'TradedShares', 'StartDate', 'EndDate', 'LaterHoldingShares'], 
                'EquityFreeze': ['UnfrozeDate', 'LegalInstitution', 'EquityHolder', 'FrozeShares', 'StartDate', 'EndDate', 'TotalHoldingShares', 'TotalHoldingRatio'], 
                'EquityPledge': ['ReleasedDate', 'Pledger', 'Pledgee', 'PledgedShares', 'StartDate', 'EndDate', 'TotalHoldingShares', 'TotalHoldingRatio', 'TotalPledgedShares'],   
                'EquityRepurchase': ['CompanyName', 'HighestTradingPrice', 'LowestTradingPrice', 'RepurchasedShares', 'ClosingDate', 'RepurchaseAmount'], 
                'NoneType': [], 
                }
        else:
            template_tokens, template_roles = None, None
        return template_tokens, template_roles

    def get_event_type2role_id_list(self, tokenizer, event_type_fields_list):
        #role embedding index
        role2id = {}
        index = 0
        event_type2role_id_list = []
        for event_type, role_type_list in event_type_fields_list:
            event_type2role_index = []
            for role_type in role_type_list:
                if role_type not in role2id.keys():
                    #we use '[unused1]' in bert vocab represent role0 and the same rule for other roles
                    role2id[role_type] = index + 2# tokenizer.convert_tokens_to_ids(['[unused%s]'%(index + 1)])[0]
                    event_type2role_index.append(index)
                    index += 1
                else:
                    event_type2role_index.append(role2id[role_type])
            event_type2role_id_list.append(event_type2role_index)
        sep = len(role2id) + 1
        role2id['Others'] = sep
        for idx, extra_role in enumerate(common_fields):
            role2id[extra_role] = sep + idx + 1
        return event_type2role_id_list, role2id

    def get_template_token_ids_tensor(self, template_tokens, event_type_fields_list, tokenizer, max_temp_len = 128, pad = '*'):
        template_token_ids_list = []
        template_key_padding_mask_list = []
        for index, (event_type, event_fields) in enumerate(event_type_fields_list):
            template_token_list = bert_chinese_char_tokenize(self.vocab, template_tokens[event_type])
            num_tokens = len(template_token_list)
            template_token_list += [pad for _ in range(max_temp_len - num_tokens)]
            template_token_ids = tokenizer.convert_tokens_to_ids(template_token_list)
            template_token_ids_list.append(template_token_ids)
            mask = [0 for _ in range(num_tokens)] + [1 for _ in range(max_temp_len - num_tokens)]
            template_key_padding_mask_list.append(mask) 

        return template_token_ids_list, template_key_padding_mask_list


    def get_role_type2template_role_index_dict(self, template_tokens, template_roles, tokenizer):
        role_type2template_role_index_dict = {}
        for event_type in template_tokens:
            role_type2template_role_index_dict[event_type] = {}
            role_types = template_roles[event_type]
            count = 0
            for index, token in enumerate(bert_chinese_char_tokenize(self.vocab, template_tokens[event_type])):
                if token == '#':
                    role_type2template_role_index_dict[event_type][role_types[count]] = index
                    count += 1
        return role_type2template_role_index_dict    
    
    def get_event_type2template_role_index_dict(self, role_type2template_index_dict, event_type_fields_list):
        '''
        {'event_type_1':[role1_index, role2_index, ...], 
        'event_type_2':[role1_index, role2_index, ...], 
        ...
        }
        '''
        event_type2template_role_index_dict = {}
        for index, (event_type, event_fields) in enumerate(event_type_fields_list):
            #print([event_type, event_fields])
            event_field2template_index = [role_type2template_index_dict[event_type][event_field] for event_field in event_fields]
            event_type2template_role_index_dict[index] = event_field2template_index
            #print({event_type: event_field2template_index})
        return event_type2template_role_index_dict

    def get_event_type2template_role_index_list(self, role_type2template_index_dict, event_type_fields_list):
        '''
        [[event_type1_id_role1_index, ...], 
        [event_type2_id_role1_index, ...], 
        ...
        ]
        '''
        event_type2template_role_index_list = []
        for index, (event_type, event_fields) in enumerate(event_type_fields_list):
            event_field2template_index = [role_type2template_index_dict[event_type][event_field] for event_field in event_fields]
            event_type2template_role_index_list.append(event_field2template_index)
            #print({event_type: event_field2template_index})
        return event_type2template_role_index_list
    
    def get_event_type2template_context_index_dict(self, template_tokens, tokenizer):
        event_type2template_context_index_dict = {}
        for event_type in template_tokens:
            event_type2template_context_index_dict[event_type] = []
            for index, token in enumerate(bert_chinese_char_tokenize(self.vocab, template_tokens[event_type])):
                if token != '#':
                    event_type2template_context_index_dict[event_type].append(index)
        return event_type2template_context_index_dict 

    def get_event_type2template_context_index_list(self, event_type2template_context_index_dict, event_type_fields_list):
        event_type2template_context_index_list = []
        for index, (event_type, event_fields) in enumerate(event_type_fields_list):
            event_type2template_context_index = event_type2template_context_index_dict[event_type]
            event_type2template_context_index_list.append(event_type2template_context_index)
        return event_type2template_context_index_list
