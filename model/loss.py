import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module): #todo should be replaced with below class
    def __init__(self, n_labels, smoothing=0.0, ignore_index=-100, size_average=True):
        super(LabelSmoothingLoss, self).__init__()
        assert 0 <= smoothing <= 1

        self.ignore_index = ignore_index
        self.confidence = 1 - smoothing

        if smoothing > 0:
            self.criterion = nn.KLDivLoss(size_average=size_average) #todo rooh: change size_average=size_average to reduction='mean'
            n_ignore_idxs = 1 + (ignore_index >= 0)
            one_hot = torch.full((1, n_labels), fill_value=(smoothing / (n_labels - n_ignore_idxs)))
            if ignore_index >= 0:
                one_hot[0, ignore_index] = 0
            self.register_buffer('one_hot', one_hot)
        else:
            #NLLLoss expects log_probs as inputs
            self.criterion = nn.NLLLoss(size_average=size_average, ignore_index=ignore_index) #todo rooh: size_average=size_average to reduction='mean'
        
    def forward(self, log_inputs, targets):
        if self.confidence < 1:#todo what's happening here?
            tdata = targets.data
  
            tmp = self.one_hot.repeat(targets.shape[0], 1)
            tmp.scatter_(1, tdata.unsqueeze(1), self.confidence)

            if self.ignore_index >= 0:
                mask = torch.nonzero(tdata.eq(self.ignore_index)).squeeze(-1)
                if mask.numel() > 0:
                    tmp.index_fill_(0, mask, 0)

            targets = tmp
        
        return self.criterion(log_inputs, targets)


class LabelSmoothingLoss2(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, vocabulary_size, pad_index=0):
        assert 0.0 < label_smoothing <= 1.0

        super().__init__()

        self.pad_index = pad_index
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.KLDivLoss(reduction='sum')

        smoothing_value = label_smoothing / (vocabulary_size - 2)  # exclude pad and true label
        smoothed_targets = torch.full((vocabulary_size,), smoothing_value)
        smoothed_targets[self.pad_index] = 0
        self.register_buffer('smoothed_targets', smoothed_targets.unsqueeze(0))  # (1, vocabulary_size)

        self.confidence = 1.0 - label_smoothing

    def forward(self, outputs, targets): #todo here we expect probs as outputs not the log
        """
        outputs (FloatTensor): (batch_size, seq_len, vocabulary_size)
        targets (LongTensor): (batch_size, seq_len)
        """
        batch_size, seq_len, vocabulary_size = outputs.size()

        outputs_log_softmax = self.log_softmax(outputs)
        outputs_flat = outputs_log_softmax.view(batch_size * seq_len, vocabulary_size)
        targets_flat = targets.view(batch_size * seq_len)

        smoothed_targets = self.smoothed_targets.repeat(targets_flat.size(0), 1)
        # smoothed_targets: (batch_size * seq_len, vocabulary_size)

        smoothed_targets.scatter_(1, targets_flat.unsqueeze(1), self.confidence)
        # smoothed_targets: (batch_size * seq_len, vocabulary_size)

        smoothed_targets.masked_fill_((targets_flat == self.pad_index).unsqueeze(1), 0)
        # masked_targets: (batch_size * seq_len, vocabulary_size)

        loss = self.criterion(outputs_flat, smoothed_targets)
        count = (targets != self.pad_index).sum().item()

        return loss, count


class RiskLoss(nn.Module):

    def __init__(self, risk_func, model, searcher, device):
        super().__init__()
        self.risk_func = risk_func
        self.model = model
        self.searcher = searcher
        self.device = device

    def forward(self, enc_contexts, targets):
        beams, beam_lengths = self.searcher.search(enc_contexts, return_beams=True)

        target_lens = targets.ne(self.model.padding_idx).sum(dim=-1)
        targets = [target[1:length - 1].tolist() for target, length in
                   zip(targets, target_lens)]  # truncate the targets to their real size without paddings

        beam_risks = []
        num_beams = beams.shape[1]
        for b in range(num_beams):
            predictions = [beam[1:l - 1].tolist() for beam, l in zip(beams[:, b, :], beam_lengths[:, b])]
            risks = torch.tensor(self.risk_func(predictions, targets), dtype=torch.float, device=self.device)
            beam_risks.append(risks)
        beam_risks = torch.stack(beam_risks, dim=-1)

        beam_probs = []
        for b in range(num_beams):
            logits = self.model.decode(beams[:, b, :-1], enc_contexts)
            logprobas = F.log_softmax(logits, dim=-1)
            logprobas = torch.gather(logprobas, -1, beams[:, b, 1:].unsqueeze(-1)).squeeze(-1) #select only beams from logprobs
            logprobas = logprobas.sum(dim=-1) / beam_lengths[:, b].float() #average over length, length normalization based on https://arxiv.org/pdf/1808.10006.pdf
            probas = logprobas.exp()
            beam_probs.append(probas)
        beam_probs = torch.stack(beam_probs, dim=-1)
        beam_probs = F.softmax(beam_probs, dim=-1)

        batch_risk_loss = torch.mean((beam_risks * beam_probs).sum(dim=-1))

        return batch_risk_loss


    # def forward(self, net_output, model, sample):
    #     """Compute the sequence-level loss for the given hypotheses.
    #
    #     Returns a tuple with three elements:
    #     1) the loss, as a Variable
    #     2) the sample size, which is used as the denominator for the gradient
    #     3) logging outputs to display while training
    #     """
    #     ######################risk on classic_seq branch ########################
    #     scores = self.get_hypothesis_scores(net_output, sample) #score are logprobs here
    #     lengths = self.get_hypothesis_lengths(net_output, sample)
    #     avg_scores = scores.sum(2) / lengths  # average over length
    #     probs = F.softmax(avg_scores.exp_()) #exponentionation makes them probs
    #     loss = (probs * sample['cost'].type_as(probs)).sum()  # average over batch
    #     sample_size = net_output.size(0)  # bsz
    #     logging_output = {
    #         'loss': loss.data[0],
    #         'sample_size': sample_size,
    #         'sum_cost': sample['cost'].sum(),
    #         'num_cost': sample['cost'].numel(),
    #     }
    #     return loss, sample_size, logging_output
    #     ####################Cross entropy on master branch#############################
    #     net_output = model(**sample['net_input'])
    #     loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
    #     sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
    #     logging_output = {
    #         'loss': utils.item(loss.data) if reduce else loss.data,
    #         'ntokens': sample['ntokens'],
    #         'nsentences': sample['target'].size(0),
    #         'sample_size': sample_size,
    #     }
    #     return loss, sample_size, logging_output
    #
    #
    #     def compute_loss(self, model, net_output, sample, reduce=True):
    #         lprobs = model.get_normalized_probs(net_output, log_probs=True)
    #         lprobs = lprobs.view(-1, lprobs.size(-1))
    #         target = model.get_targets(sample, net_output).view(-1)
    #         loss = F.nll_loss(lprobs, target, size_average=False, ignore_index=self.padding_idx,
    #                           reduce=reduce)
    #         return loss, loss
