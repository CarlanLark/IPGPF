# -*- coding: utf-8 -*-


class BaseEvent(object):
    def __init__(self, fields, event_name='Event', key_fields=(), recguid=None):
        self.recguid = recguid
        self.name = event_name
        self.fields = list(fields)
        self.field2content = {f: None for f in fields}
        self.nonempty_count = 0
        self.nonempty_ratio = self.nonempty_count / len(self.fields)

        self.key_fields = set(key_fields)
        for key_field in self.key_fields:
            assert key_field in self.field2content

    def __repr__(self):
        event_str = "\n{}[\n".format(self.name)
        event_str += "  {}={}\n".format("recguid", self.recguid)
        event_str += "  {}={}\n".format("nonempty_count", self.nonempty_count)
        event_str += "  {}={:.3f}\n".format("nonempty_ratio", self.nonempty_ratio)
        event_str += "] (\n"
        for field in self.fields:
            if field in self.key_fields:
                key_str = " (key)"
            else:
                key_str = ""
            event_str += "  " + field + "=" + str(self.field2content[field]) + ", {}\n".format(key_str)
        event_str += ")\n"
        return event_str

    def update_by_dict(self, field2text, recguid=None):
        self.nonempty_count = 0
        self.recguid = recguid

        for field in self.fields:
            if field in field2text and field2text[field] is not None:
                self.nonempty_count += 1
                self.field2content[field] = field2text[field]
            else:
                self.field2content[field] = None

        self.nonempty_ratio = self.nonempty_count / len(self.fields)

    def field_to_dict(self):
        return dict(self.field2content)

    def set_key_fields(self, key_fields):
        self.key_fields = set(key_fields)

    def is_key_complete(self):
        for key_field in self.key_fields:
            if self.field2content[key_field] is None:
                return False

        return True

    def is_good_candidate(self):
        raise NotImplementedError()

    def get_argument_tuple(self):
        args_tuple = tuple(self.field2content[field] for field in self.fields)
        return args_tuple


class EquityFreezeEvent(BaseEvent):
    NAME = 'EquityFreeze'
    FIELDS = [
        'EquityHolder',
        'FrozeShares',
        'LegalInstitution',
        'TotalHoldingShares',
        'TotalHoldingRatio',
        'StartDate',
        'EndDate',
        'UnfrozeDate',
    ]

    def __init__(self, recguid=None):
        super().__init__(
            EquityFreezeEvent.FIELDS, event_name=EquityFreezeEvent.NAME, recguid=recguid
        )
        self.set_key_fields([
            'EquityHolder',
            'FrozeShares',
            'LegalInstitution',
        ])

    def is_good_candidate(self, min_match_count=5):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


class EquityRepurchaseEvent(BaseEvent):
    NAME = 'EquityRepurchase'
    FIELDS = [
        'CompanyName',
        'HighestTradingPrice',
        'LowestTradingPrice',
        'RepurchasedShares',
        'ClosingDate',
        'RepurchaseAmount',
    ]

    def __init__(self, recguid=None):
        super().__init__(
            EquityRepurchaseEvent.FIELDS, event_name=EquityRepurchaseEvent.NAME, recguid=recguid
        )
        self.set_key_fields([
            'CompanyName',
        ])

    def is_good_candidate(self, min_match_count=4):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


class EquityUnderweightEvent(BaseEvent):
    NAME = 'EquityUnderweight'
    FIELDS = [
        'EquityHolder',
        'TradedShares',
        'StartDate',
        'EndDate',
        'LaterHoldingShares',
        'AveragePrice',
    ]

    def __init__(self, recguid=None):
        super().__init__(
            EquityUnderweightEvent.FIELDS, event_name=EquityUnderweightEvent.NAME, recguid=recguid
        )
        self.set_key_fields([
            'EquityHolder',
            'TradedShares',
        ])

    def is_good_candidate(self, min_match_count=4):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


class EquityOverweightEvent(BaseEvent):
    NAME = 'EquityOverweight'
    FIELDS = [
        'EquityHolder',
        'TradedShares',
        'StartDate',
        'EndDate',
        'LaterHoldingShares',
        'AveragePrice',
    ]

    def __init__(self, recguid=None):
        super().__init__(
            EquityOverweightEvent.FIELDS, event_name=EquityOverweightEvent.NAME, recguid=recguid
        )
        self.set_key_fields([
            'EquityHolder',
            'TradedShares',
        ])

    def is_good_candidate(self, min_match_count=4):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


class EquityPledgeEvent(BaseEvent):
    NAME = 'EquityPledge'
    FIELDS = [
        'Pledger',
        'PledgedShares',
        'Pledgee',
        'TotalHoldingShares',
        'TotalHoldingRatio',
        'TotalPledgedShares',
        'StartDate',
        'EndDate',
        'ReleasedDate',
    ]

    def __init__(self, recguid=None):
        # super(EquityPledgeEvent, self).__init__(
        super().__init__(
            EquityPledgeEvent.FIELDS, event_name=EquityPledgeEvent.NAME, recguid=recguid
        )
        self.set_key_fields([
            'Pledger',
            'PledgedShares',
            'Pledgee',
        ])

    def is_good_candidate(self, min_match_count=5):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False

class EquityPledgeEvent2(BaseEvent):
    NAME = "质押"
    FIELDS = [
        "质押物占总股比",
        "质权方",
        "质押方",
        "事件时间",
        "质押股票/股份数量",
        "质押物所属公司",
        "质押物",
        "质押物占持股比",
        "披露时间",
    ]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.FIELDS)


class ShareRepurchaseEvent(BaseEvent):
    NAME = "股份回购"
    FIELDS = ["每股交易价格", "交易金额", "回购完成时间", "回购股份数量", "占公司总股本比例", "回购方", "披露时间"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.FIELDS)


class ReleasePledgeEvent(BaseEvent):
    NAME = "解除质押"
    FIELDS = [
        "质权方",
        "质押物占总股比",
        "质押方",
        "事件时间",
        "质押股票/股份数量",
        "质押物所属公司",
        "质押物",
        "质押物占持股比",
        "披露时间",
    ]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.FIELDS)


class InvitedConversationEvent(BaseEvent):
    NAME = "被约谈"
    FIELDS = ["约谈机构", "被约谈时间", "披露时间", "公司名称"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.FIELDS)


class BusinessAcquisitionEvent(BaseEvent):
    NAME = "企业收购"
    FIELDS = ["被收购方", "收购标的", "交易金额", "收购方", "收购完成时间", "披露时间"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.FIELDS)


class ShareholderOverweightEvent(BaseEvent):
    NAME = "股东增持"
    FIELDS = [
        "每股交易价格",
        "交易金额",
        "增持部分占所持比例",
        "交易完成时间",
        "增持方",
        "交易股票/股份数量",
        "增持部分占总股本比例",
        "股票简称",
        "披露时间",
    ]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.FIELDS)


class ExecutivesChangeEvent(BaseEvent):
    NAME = "高管变动"
    FIELDS = ["变动后职位", "任职公司", "高管姓名", "披露日期", "变动类型", "事件时间", "高管职位", "变动后公司名称"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.FIELDS)


class WinBidEvent(BaseEvent):
    NAME = "中标"
    FIELDS = ["中标金额", "披露日期", "招标方", "中标日期", "中标标的", "中标公司"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.FIELDS)


class CompanyIPOEvent(BaseEvent):
    NAME = "公司上市"
    FIELDS = ["募资金额", "事件时间", "证券代码", "环节", "发行价格", "上市公司", "披露时间", "市值"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.FIELDS)


class CompanyFinancingEvent(BaseEvent):
    NAME = "企业融资"
    FIELDS = ["融资金额", "事件时间", "被投资方", "领投方", "融资轮次", "披露时间", "投资方"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.FIELDS)


class CompanyLossEvent(BaseEvent):
    NAME = "亏损"
    FIELDS = ["亏损变化", "财报周期", "净亏损", "披露时间", "公司名称"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.FIELDS)


class ShareholderUnderweightEvent(BaseEvent):
    NAME = "股东减持"
    FIELDS = [
        "减持方",
        "每股交易价格",
        "交易金额",
        "减持部分占所持比例",
        "交易完成时间",
        "交易股票/股份数量",
        "减持部分占总股本比例",
        "股票简称",
        "披露时间",
    ]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.FIELDS)


class CompanyBankruptEvent(BaseEvent):
    NAME = "企业破产"
    FIELDS = ["债务规模", "破产公司", "债权人", "破产时间", "披露时间"]

    def __init__(self, recguid=None):
        super().__init__(self.FIELDS, event_name=self.NAME, recguid=recguid)
        self.set_key_fields(self.FIELDS)


dataset = ['ChFinAnn', 'DuEE-fin'][1]
merge_entity = [True, False][0]
if dataset == 'ChFinAnn':
    event_type2event_class = {
        EquityFreezeEvent.NAME: EquityFreezeEvent,
        EquityRepurchaseEvent.NAME: EquityRepurchaseEvent,
        EquityUnderweightEvent.NAME: EquityUnderweightEvent,
        EquityOverweightEvent.NAME: EquityOverweightEvent,
        EquityPledgeEvent.NAME: EquityPledgeEvent,
    }
    event_type_fields_list = [
        (EquityFreezeEvent.NAME, EquityFreezeEvent.FIELDS),
        (EquityRepurchaseEvent.NAME, EquityRepurchaseEvent.FIELDS),
        (EquityUnderweightEvent.NAME, EquityUnderweightEvent.FIELDS),
        (EquityOverweightEvent.NAME, EquityOverweightEvent.FIELDS),
        (EquityPledgeEvent.NAME, EquityPledgeEvent.FIELDS),
    ]
    if merge_entity:
        common_fields = ['StockCode', 'StockAbbr', 'CompanyName', 'ratio', 'money', 'date', 'share', 'holder']
        local_type2global_type = {
                # shares
                "TotalHoldingShares": "share",
                "TotalPledgedShares": "share",
                "PledgedShares": "share",
                "FrozeShares": "share",
                "RepurchasedShares": "share",
                "TradedShares": "share",
                "LaterHoldingShares": "share",
                # ratio
                "TotalHoldingRatio": "ratio",
                # date
                "StartDate": "date",
                "ReleasedDate": "date",
                "EndDate": "date",
                "ClosingDate": "date",
                "UnfrozeDate": "date",
                # money
                "RepurchaseAmount": "money",
                "HighestTradingPrice": "money",
                "LowestTradingPrice": "money",
                "AveragePrice": "money",
                # holder
                "EquityHolder": "holder", 
                "CompanyName": "holder", 
                "Pledger": "holder", 
                "Pledgee": "holder", 
                }
        global_type2local_type = {
                        'share': ['TotalHoldingShares', 'TotalPledgedShares', 'PledgedShares', 'FrozeShares', 'RepurchasedShares', 'TradedShares', 'LaterHoldingShares', ], 
                        'ratio': ['TotalHoldingRatio'], 
                        'date': ['StartDate', 'ReleasedDate', 'EndDate', 'ClosingDate', 'UnfrozeDate', ], 
                        'money': ['RepurchaseAmount', 'HighestTradingPrice', 'LowestTradingPrice', 'AveragePrice', ], 
                        'holder': ['EquityHolder', 'CompanyName', 'Pledger', 'Pledgee', ], 
                        }
    else:
        common_fields = ['StockCode', 'StockAbbr', 'CompanyName']
        local_type2global_type = {}
        global_type2local_type = {}

elif dataset == 'DuEE-fin':
    event_type2event_class = {
        EquityPledgeEvent2.NAME: EquityPledgeEvent2,
        ShareRepurchaseEvent.NAME: ShareRepurchaseEvent,
        ReleasePledgeEvent.NAME: ReleasePledgeEvent,
        InvitedConversationEvent.NAME: InvitedConversationEvent,
        BusinessAcquisitionEvent.NAME: BusinessAcquisitionEvent,
        ShareholderOverweightEvent.NAME: ShareholderOverweightEvent,
        ExecutivesChangeEvent.NAME: ExecutivesChangeEvent,
        WinBidEvent.NAME: WinBidEvent,
        CompanyIPOEvent.NAME: CompanyIPOEvent,
        CompanyFinancingEvent.NAME: CompanyFinancingEvent,
        CompanyLossEvent.NAME: CompanyLossEvent,
        ShareholderUnderweightEvent.NAME: ShareholderUnderweightEvent,
        CompanyBankruptEvent.NAME: CompanyBankruptEvent,
    }

    event_type_fields_list = [
        # name, fields
        (EquityPledgeEvent2.NAME, EquityPledgeEvent2.FIELDS),
        (ShareRepurchaseEvent.NAME, ShareRepurchaseEvent.FIELDS),
        (ReleasePledgeEvent.NAME, ReleasePledgeEvent.FIELDS),
        (InvitedConversationEvent.NAME, InvitedConversationEvent.FIELDS),
        (BusinessAcquisitionEvent.NAME, BusinessAcquisitionEvent.FIELDS),
        (ShareholderOverweightEvent.NAME, ShareholderOverweightEvent.FIELDS),
        (ExecutivesChangeEvent.NAME, ExecutivesChangeEvent.FIELDS),
        (WinBidEvent.NAME, WinBidEvent.FIELDS),
        (CompanyIPOEvent.NAME, CompanyIPOEvent.FIELDS),
        (CompanyFinancingEvent.NAME, CompanyFinancingEvent.FIELDS),
        (CompanyLossEvent.NAME, CompanyLossEvent.FIELDS),
        (ShareholderUnderweightEvent.NAME, ShareholderUnderweightEvent.FIELDS),
        (CompanyBankruptEvent.NAME, CompanyBankruptEvent.FIELDS),
    ]
    common_fields = []
    local_type2global_type = {}
    global_type2local_type = {}