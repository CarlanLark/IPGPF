# -*- coding: utf-8 -*-

import json
import logging
import pickle
import re

from collections import defaultdict
from typing import List, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-10


def default_load_json(json_file_path, encoding='utf-8', **kwargs):
    with open(json_file_path, 'r', encoding=encoding) as fin:
        tmp_json = json.load(fin, **kwargs)
    return tmp_json


def default_dump_json(obj, json_file_path, encoding='utf-8', ensure_ascii=False, indent=2, **kwargs):
    with open(json_file_path, 'w', encoding=encoding) as fout:
        json.dump(obj, fout,
                  ensure_ascii=ensure_ascii,
                  indent=indent,
                  **kwargs)


def default_load_pkl(pkl_file_path, **kwargs):
    with open(pkl_file_path, 'rb') as fin:
        obj = pickle.load(fin, **kwargs)

    return obj


def default_dump_pkl(obj, pkl_file_path, **kwargs):
    with open(pkl_file_path, 'wb') as fout:
        pickle.dump(obj, fout, **kwargs)


def set_basic_log_config(log_path):
    logging.basicConfig(filename = log_path, 
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)



def bert_chinese_char_tokenize(vocab, text, unk_token='[UNK]'):
    """perform pure character-based tokenization"""
    tokens = list(text)
    out_tokens = []
    for token in tokens:
        if token in vocab:
            out_tokens.append(token)
        else:
            out_tokens.append(unk_token)

    return out_tokens

def recursive_print_grad_fn(grad_fn, prefix='', depth=0, max_depth=50):
    if depth > max_depth:
        return
    print(prefix, depth, grad_fn.__class__.__name__)
    if hasattr(grad_fn, 'next_functions'):
        for nf in grad_fn.next_functions:
            ngfn = nf[0]
            recursive_print_grad_fn(ngfn, prefix=prefix + '  ', depth=depth+1, max_depth=max_depth)


def strtobool(str_val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    str_val = str_val.lower()
    if str_val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif str_val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (str_val,))


def match_arg(
    sentences: List[str],
    doc_token_ids: np.ndarray,
    arg: Tuple[int],
    offset: Optional[int] = 0,
):
    arg_arr = np.array(arg)
    arg_len = len(arg)
    tokens_ravel = np.ravel(doc_token_ids)
    seq_len = doc_token_ids.shape[1]
    row = col = None
    for i in range(0, tokens_ravel.shape[0] - arg_arr.shape[0] + 1):
        if np.array_equal(arg_arr, tokens_ravel[i : i + arg_len]):
            row = i // seq_len
            col = i - row * seq_len
    if row is not None and col is not None:
        return "".join(sentences[row][col - offset : col + arg_len - offset]), [
            row,
            col - offset,
            col + arg_len - offset,
        ]
    else:
        return None, None


class RegexEntExtractor(object): # modified from https://github.com/Spico197/DocEE
    def __init__(self) -> None:
        self.field2type = {
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
            # shares
            "质押股票/股份数量": "share",
            "回购股份数量": "share",
            "交易股票/股份数量": "share",
            # ratio
            "质押物占总股比": "ratio",
            "质押物占持股比": "ratio",
            "占公司总股本比例": "ratio",
            "增持部分占总股本比例": "ratio",
            "增持部分占所持比例": "ratio",
            "减持部分占总股本比例": "ratio",
            "减持部分占所持比例": "ratio",
            # date
            "披露时间": "date",
            "披露日期": "date",
            "中标日期": "date",
            "事件时间": "date",
            "回购完成时间": "date",
            "被约谈时间": "date",
            "收购完成时间": "date",
            "交易完成时间": "date",
            "破产时间": "date",
            # money
            "每股交易价格": "money",
            "交易金额": "money",
            "募资金额": "money",
            "发行价格": "money",
            "市值": "money",
            "融资金额": "money",
            "净亏损": "money",
        }
        self.field_id2field_name = {}
        self.basic_type_id = None  # id of `O` label
        self.type2func = {
            "share": self.extract_share,
            "ratio": self.extract_ratio,
            "date": self.extract_date,
            "money": self.extract_money,
        }

    @classmethod
    def _extract(cls, regex, text, group=0):
        results = []
        matches = re.finditer(regex, text)
        for match in matches:
            results.append([match.group(group), match.span(group)])
        return results

    @classmethod
    def extract_share(cls, text):
        regex = r"(\d+股)[^票]"
        results = cls._extract(regex, text, group=1)
        return results

    @classmethod
    def extract_ratio(cls, text):
        regex = r"\d+(\.\d+)?%"
        results = cls._extract(regex, text)
        return results

    @classmethod
    def extract_date(cls, text):
        regex = r"\d{4}年\d{1,2}月\d{1,2}日"
        results = cls._extract(regex, text)
        return results

    @classmethod
    def extract_money(cls, text):
        regex = r"\d+(\.\d+)?元"
        results = cls._extract(regex, text)
        return results

    def extract(self, text):
        r"""
        extract ents from one sentence
        Returns:
            {
                "ratio": [[ent, (start pos, end pos)], ...],
                ...
            }
        """
        field2results = defaultdict(list)
        for field, func in self.type2func.items():
            results = func(text)
            if len(results) > 0:
                field2results[field].extend(results)
        return field2results

    def extract_doc(
        self, doc: List[str], exclude_ents: Optional[List[str]] = []
    ) -> Dict[str, List]:
        r"""
        extract ents from the whole document (multiple lines)
        Returns:
            {
                "ratio": [[ent, (sentence idx, start pos, end pos)], ...],
                ...
            }
        """
        field2results = defaultdict(list)
        for sent_idx, line in enumerate(doc):
            results = self.extract(line)
            for field, fr in results.items():
                for match_text, match_span in fr:
                    if match_text not in exclude_ents:
                        field2results[field].append(
                            [match_text, [sent_idx, match_span[0], match_span[1]]]
                        )
        return field2results


regex_extractor = RegexEntExtractor()