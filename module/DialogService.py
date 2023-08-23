import os
import numpy as np
import torch
import re
import argparse

from module.dialog.kogpt2 import DialogKoGPT2
from kogpt2_transformers import get_kogpt2_tokenizer

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from module.dialog.collate import PREPARE, COLLATE
from torch.nn.functional import softmax
from torch.nn.modules import ModuleList, Module
from torch.utils.checkpoint import checkpoint
import torch

import logging
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

import gensim # 3.8.1
import time
import math
from konlpy.tag import Mecab
mecab = Mecab()
w2v = gensim.models.Word2Vec.load("../circulus-napi-model/vector/ko/vector.bin")
cnt_return = 3

def process(sentence):
    items = []
    for item in mecab.pos(sentence):
        if item[1].startswith('N') or item[1].startswith('V') or item[1].startswith('M'):
            if item[1].startswith('V'):
                items.append(f"{item[0]}다")
            elif len(item[0]) ==  1:
                items.append(item[0])
            else:
                items.append(item[0])
    #print(items)            
    return items

parser = argparse.ArgumentParser(description='MentalHealth-bot based on KoGPT-2')

parser.add_argument('--model_params',
                    type=str,
                    default='model_chp/model_-last.ckpt',
                    help='model binary for starting chat')

parser.add_argument('--train',
                    action='store_true',
                    default=False,
                    help='for training')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

U_TKN = '<usr>'
S_TKN = '<sys>'
EOS = '</s>'
MASK = '<unused0>'
PAD = '<pad>'
SENT = '<unused1>'
UNK = '<unk>'

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 

parser = argparse.ArgumentParser(description='pibot based on KoGPT-2')

parser.add_argument('--chat',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')

args = parser.parse_args()


class KoGPT2Chat(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2Chat, self).__init__()
        #self.hparams = hparams
        self.save_hyperparameters(hparams)
        #self.hparams.update(hparams)
        self.neg = -1e18
        self.kogpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len',
                            type=int,
                            default=48,
                            help='max sentence length on input (default: 32)')

        parser.add_argument('--batch-size',
                            type=int,
                            default=144,
                            help='batch size for training (default: 96)')
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        return parser

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output = self.kogpt2(inputs, return_dict=True)
        return output.logits

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        self.log('train_loss', loss_avg)
        return loss_avg

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        #data = pd.read_csv('chatbot_dataset.csv')
        data = pd.read_json('./input/data.json')
        self.train_set = CharDataset(data, max_len=self.hparams.max_len)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=4,
            shuffle=True, collate_fn=self._collate_fn)
        return train_dataloader



device_ko = torch.device("cuda:0")
output_size = 40 

"""
checkpoint_ko = torch.load("../circulus-napi-model/dialog/ko/kogpt2-wellnesee-auto-regressive.pth", map_location=device_ko)
model_ko = DialogKoGPT2()
model_ko.load_state_dict(checkpoint_ko['model_state_dict'])
model_ko.eval()
model_ko.to(device_ko)
tokenizer_ko = get_kogpt2_tokenizer()
"""

model = KoGPT2Chat(args)
model = model.load_from_checkpoint("../circulus-napi-model/dialog/ko/model_-last.ckpt")
model.to(device_ko)
model.eval()


def predict_dialog(text : str, lang : str):
    
    answer = ''
    user = U_TKN + text + SENT + answer
    encoded = tokenizer.encode(user)
    input_ids = torch.LongTensor(encoded).unsqueeze(dim=0).to(device_ko)
    bad_words_ids = [tokenizer(bad_word).input_ids for bad_word in ["여자","남자","아내","남편","자식","자녀","그녀"]]

    # https://huggingface.co/blog/how-to-generate
    # https://huggingface.co/transformers/main_classes/model.html?highlight=generate
    output = model.kogpt2.generate(input_ids, 
        min_length=24, max_length=48, 
        #length_penalty=1.4, repetition_penalty=1.4,
        top_k=30, top_p=0.9, temperature=0.9, 
        do_sample= True, early_stopping=True,
        #num_beams=1, early_stopping=False, no_repeat_ngram_size=None, 
        #max_time = 1.9, bad_words_ids=bad_words_ids, 
        num_return_sequences=cnt_return)
    """
    https://littlefoxdiary.tistory.com/4
    
    output = model.kogpt2.generate(input_ids, 
        min_length=26, max_length=48, 
        length_penalty=1.4, repetition_penalty=1.4,
        top_k=24, top_p=0.9, temperature=0.8, 
        do_sample= True, early_stopping=True,
        #num_beams=1, early_stopping=False, no_repeat_ngram_size=None, 
        #max_time = 1.9, bad_words_ids=bad_words_ids, 
        num_return_sequences=cnt_return)

    sample_outputs = model.generate(
        input_ids,
        do_sample=True, #샘플링 전략 사용
        max_length=50, # 최대 디코딩 길이는 50
        top_k=50, # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
        top_p=0.95, # 누적 확률이 95%인 후보집합에서만 생성
        num_return_sequences=3 #3개의 결과를 디코딩해낸다
    )

    tokenized_indexs = tokenizer_ko.encode(text)
    input_ids = torch.tensor([tokenizer_ko.bos_token_id,]  + tokenized_indexs + [tokenizer_ko.eos_token_id]).unsqueeze(0)
    input_ids = input_ids.to(device_ko)
    # set top_k to 50
    sample_output = model_ko.generate(input_ids=input_ids, max_length=output_size, pad_token_id=tokenizer_ko.pad_token_id)
    candidate = tokenizer_ko.decode(sample_output[0].tolist()[len(tokenized_indexs)+1:],skip_special_tokens=True)
    candidate = candidate.replace("?",".").replace("!",".")
    candidate = candidate.split(".")
    """
    answers = []
    question = text

    #if len(q_history) > 0:
    #    question = f"{q_history[-1]} {q}"

    proc_q = process(question)
    scores = []

    results = []

    cnt = 0

    for _ in range(cnt_return):
        idx = torch.where(output[cnt]==tokenizer.encode('<sys>')[0])
        print(idx)
        answer = tokenizer.decode(output[cnt][int(idx[0])+1:], skip_special_tokens=True)
        proc_a = process(answer)
        distance = round(w2v.wmdistance(proc_q,proc_a),3)
        #distance2 = w2v.n_similarity(process(q), process(answer))
        if math.isinf(distance):
            distance = -1
        answers.append(answer)
        scores.append(distance)
        #print(f"{cnt} {answer} {distance}")

        results.append({ "answer" : answer, "score" : distance})

        cnt = cnt + 1

    max_value = max(scores)
    min_value = min(scores)

    index = scores.index(max_value)

    if max_value < 20: # too far or not special, closer value is better
        if min_value < 10:
            # good luck mode
            for _ in range(cnt_return):
                if scores[_] != min_value and scores[_] != max_value:
                    index = _
                    break
        else:
            index = scores.index(min_value)
    elif max_value > 22:
        for _ in range(cnt_return):
            if scores[_] != min_value and scores[_] != max_value:
                index = _
                break

    return results#answers[index]
