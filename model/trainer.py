import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import f1_score, bleu_score
from utils.text import BPEVocab
from utils.common import pad_sequence
from model.optim import Adam, NoamOpt
from model.loss import LabelSmoothingLoss, RiskLoss


class Trainer:
    def __init__(self, model, searcher, train_dataloader, test_dataloader=None, optimizer=None, batch_size=8,
                 batch_split=1, lm_weight=0.5, risk_weight=0, lr=6.25e-5, lr_warmup=2000,
                 test_period=1, clip_grad=None, label_smoothing=0, last_checkpoint_path=None, device=torch.device('cuda'),
                 vocab=None):

        self.model = model.to(device)
        self.searcher = searcher
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=self.model.padding_idx).to(device)
        self.criterion = LabelSmoothingLoss(n_labels=self.model.n_embeddings, smoothing=label_smoothing,
                                            ignore_index=self.model.padding_idx).to(device)
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        if test_dataloader is not None:
            self.test_dataloader = test_dataloader

        self.batch_split = batch_split
        self.lm_weight = lm_weight
        self.risk_weight = risk_weight
        self.clip_grad = clip_grad
        self.device = device
        self.ignore_idxs = vocab.special_tokens_ids # for special characters <UNK>, <bos>, ...total=8
        self.vocab = vocab
        self.test_period = test_period
        self.last_checkpoint_path = last_checkpoint_path

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        if 'model' in state_dict:
            self.model.load_state_dict(state_dict['model'], strict=False)
            self.optimizer.load_state_dict(state_dict['optimizer'])
        else:
            self.model.load_state_dict(state_dict)

    def train_epoch(self, epoch, data, risk_func=None, experiment=None):
        self.model.train()  # set the model to training mode(this is not training)

        #tqdm.monitor_interval = 0
        #tqdm_data = tqdm(self.train_dataloader, desc='Train (epoch #{})'.format(epoch))
        loss = 0
        lm_loss = 0
        risk_loss = 0

        for i, (contexts, targets) in experiment.batch_loop(iterable=self.train_dataloader):
            contexts, targets = [c.to(self.device) for c in contexts], targets.to(self.device)

            enc_contexts = []

            # lm loss
            batch_lm_loss = torch.tensor(0, dtype=torch.float, device=self.device)
            for context in contexts:
                enc_context = self.model.encode(context.clone())#looks like we need clone for cuda 10(not needed for cuda 9)
                enc_contexts.append(enc_context)

                if self.lm_weight > 0:
                    context_logits = self.model.generate(enc_context[0])  #we select the zero element because it's the real encoding, the enc_context[1] is padding_mask gives presoftmax weights
                    ignore_mask = torch.stack([context == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1)
                    context = context.masked_fill_(ignore_mask, self.model.padding_idx) #this line and the previous one removes the special characters
                    prevs, nexts = context_logits[:, :-1, :].contiguous(), context[:, 1:].contiguous()
                    batch_lm_loss += (self.lm_criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1)) / len(contexts))

            # s2s loss
            prevs, nexts = targets[:, :-1].contiguous(), targets[:, 1:].contiguous()
            outputs = self.model.decode(prevs, enc_contexts) # this gives the pre-softmax todo: rooh, write a softmax function in model
            outputs = F.log_softmax(outputs, dim=-1)
            batch_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))

            # risk loss
            batch_risk_loss = torch.tensor(0, dtype=torch.float, device=self.device)
            if risk_func is not None and self.risk_weight > 0:
                risk_criterion = RiskLoss(risk_func, self.model, self.searcher, self.device)
                batch_risk_loss = risk_criterion(enc_contexts, targets)

            # optimization
            full_loss = (batch_lm_loss * self.lm_weight + self.risk_weight * batch_risk_loss + batch_loss) / self.batch_split
            full_loss.backward()

            if (i + 1) % self.batch_split == 0:
                if self.clip_grad is not None:
                    for group in self.optimizer.param_groups:
                        nn.utils.clip_grad_norm_(group['params'], self.clip_grad)
                self.optimizer.step()
                self.optimizer.zero_grad()

            lm_loss = (i * lm_loss + batch_lm_loss.item()) / (i + 1) #online average
            loss = (i * loss + batch_loss.item()) / (i + 1)
            risk_loss = (i * risk_loss + batch_risk_loss.item()) / (i + 1)

            #tqdm_data.set_postfix({'lm_loss': lm_loss, 'loss': loss, 'risk_loss': risk_loss})
            print(str({'i': str(i), 'lm_loss': lm_loss, 'loss': loss, 'risk_loss': risk_loss}))

            if i % 10 == 0:
                experiment.add_metric("Loss", loss)
                experiment.add_metric("LM_Loss", lm_loss)
                experiment.add_metric("Risk", risk_loss)


    def test_epoch(self, metric_funcs={}):
        self.model.eval()

        tqdm_data = tqdm(self.test_dataloader, desc='Test')
        loss = 0
        lm_loss = 0
        metrics = {name: 0 for name in metric_funcs.keys()}
        for i, (contexts, targets) in enumerate(tqdm_data):
            contexts, targets = [c.to(self.device) for c in contexts], targets.to(self.device)

            enc_contexts = []

            # lm loss
            batch_lm_loss = torch.tensor(0, dtype=torch.float, device=self.device)
            for context in contexts:
                enc_context = self.model.encode(context.clone())
                enc_contexts.append(enc_context)

                if self.lm_weight > 0:
                    context_outputs = self.model.generate(enc_context[0])
                    ignore_mask = torch.stack([context == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1)
                    context.masked_fill_(ignore_mask, self.model.padding_idx)
                    prevs, nexts = context_outputs[:, :-1, :].contiguous(), context[:, 1:].contiguous()
                    batch_lm_loss += (self.lm_criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1)) / len(contexts))

            # s2s loss
            prevs, nexts = targets[:, :-1].contiguous(), targets[:, 1:].contiguous()
            outputs = self.model.decode(prevs, enc_contexts)
            outputs = F.log_softmax(outputs, dim=-1)
            batch_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))
            #new compared to eval_train
            predictions = self.searcher.predict(enc_contexts)
            target_lens = targets.ne(self.model.padding_idx).sum(dim=-1)
            targets = [t[1:l - 1].tolist() for t, l in
                       zip(targets, target_lens)]  # truncate the targets to their real size without paddings

            lm_loss = (i * lm_loss + batch_lm_loss.item()) / (i + 1)
            loss = (i * loss + batch_loss.item()) / (i + 1)
            for name, func in metric_funcs.items():
                score = func(predictions, targets)
                metrics[name] = (metrics[name] * i + score) / (i + 1)

            tqdm_data.set_postfix(dict({'lm_loss': lm_loss, 'loss': loss}, **metrics))

    def test(self, metric_funcs={}):
        if hasattr(self, 'test_dataloader'):
            self.test_epoch(metric_funcs)

    def train(self, epochs, after_epoch_funcs=[], risk_func=None, experiment=None):
        #after_epoch_funcs = [self.sample_text_func, self.test_func, self.save_func]

        #sanity check to make sure there is no problem with the model
        #first_batch = next(iter(self.train_dataloader))

        after_epoch_funcs = [self.save_func]
        for epoch in experiment.epoch_loop(epochs):
            #data = [([first_batch[0][0].clone(), first_batch[0][1].clone()], first_batch[1].clone()) for _ in range(500)]
            self.train_epoch(epoch, [], risk_func, experiment)

            for func in after_epoch_funcs:
                func(epoch)

    def sample_text_func(self, epoch):
        sample_size = 5
        test_dataset = [next(iter(self.test_dataloader)) for _ in range(sample_size)] #todo this should be random

        n_samples = 5
        samples_idxs = random.sample(range(len(test_dataset)), n_samples)
        samples = [test_dataset[idx] for idx in samples_idxs]
        for persona_info, dialog, target in samples:
            contexts = [torch.tensor([c], dtype=torch.long, device=self.device) for c in [persona_info, dialog]
                        if len(c) > 0]

            #prediction = self.model.predict(contexts)[0]
            prediction = self.searcher.predict(contexts)[0]

            persona_info_str = self.vocab.ids2string(persona_info[1:-1])
            dialog_str = self.vocab.ids2string(dialog)
            dialog_str = dialog_str.replace(self.vocab.talker1_bos, '\n\t- ').replace(self.vocab.talker2_bos, '\n\t- ')
            dialog_str = dialog_str.replace(self.vocab.talker1_eos, '').replace(self.vocab.talker2_eos, '')
            target_str = self.vocab.ids2string(target[1:-1])
            prediction_str = self.vocab.ids2string(prediction)

            print('\n')
            print('Persona info:\n\t{}'.format(persona_info_str))
            print('Dialog:{}'.format(dialog_str))
            print('Target:\n\t{}'.format(target_str))
            print('Prediction:\n\t{}'.format(prediction_str))

    def test_func(self, epoch):
        if (epoch + 1) % self.trainer_config.test_period == 0:
            metric_funcs = {'f1_score': f1_score, "bleu_score": bleu_score}
            self.model.test(metric_funcs)

    def save_func(self, epoch):
        torch.save(self.state_dict(), self.last_checkpoint_path)
        print("model saved for epoch ", epoch)