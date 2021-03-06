import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.search import BeamSearch

from model.transformer_module import TransformerModule, TransformerLMHead


class TransformerModel(nn.Module):
    def __init__(self, n_layers, n_embeddings, n_pos_embeddings, embeddings_size, 
                 padding_idx, n_heads, dropout, embed_dropout, attn_dropout, ff_dropout,
                 bos_id, eos_id, max_seq_len=256, beam_size=5, sample=False,
                 length_penalty=0.8, annealing_topk=None, annealing=0, 
                 diversity_coef=0, diversity_groups=1, n_segments=None):

        super(TransformerModel, self).__init__()

        self.padding_idx = padding_idx
        self.n_embeddings = n_embeddings
        self.n_pos_embeddings = n_pos_embeddings
        self.embeddings_size = embeddings_size

        self.bos_id = bos_id
        self.eos_id = eos_id

        # self.max_seq_len = max_seq_len
        # self.beam_size = beam_size
        # self.sample = sample
        # self.length_penalty_coef = length_penalty
        # self.annealing = annealing
        # self.annealing_topk = annealing_topk
        # self.diversity_coef = diversity_coef
        # self.diversity_groups = diversity_groups

        self.transformer_module = TransformerModule(n_layers, n_embeddings, n_pos_embeddings, embeddings_size, 
                                                    padding_idx, n_heads, dropout, embed_dropout, attn_dropout,
                                                    ff_dropout, n_segments)

        #probably we don't need these if we add TransformerLMHead###########
        self.decoder = nn.Linear(embeddings_size, n_embeddings, bias=False) 
        # Share the weight matrix between word embedding & the final logit dense layer(pre_softmax)
        self.decoder.weight = self.transformer_module.embeddings.weight
        ########################################

        # todo need to be added if we want to use TransformerLMHead
        #self.lm_head = TransformerLMHead(self.transformer_module.embeddings.weight)

    def forward(self, x, contexts=[]):
        enc_contexts = [self.encode(c) for c in contexts]
        return self.decode(x, enc_contexts)

    def encode(self, x):
        return self.transformer_module(x)

    def generate(self, enc_x):
        return self.decoder(enc_x)

    def decode(self, x, enc_contexts=[]):
        x, _ = self.transformer_module(x, enc_contexts)
        return self.generate(x)

    #todo need to be added if we want to use TransformerLMHead

    # def forward(self, contexts, targets):
    #     # lm loss
    #     enc_contexts = []
    #     batch_lm_loss = torch.tensor(0, dtype=torch.float, device=self.device)
    #     for context in contexts:
    #         enc_context = self.encode(context.clone())  # looks like we need clone for cuda 10(not needed for cuda 9)
    #         enc_contexts.append(enc_context)
    #
    #         if self.lm_weight > 0:
    #             lm_logits = self.lm_head(enc_contexts[0]) # we select the zero element because it's the real encoding, the enc_context[1] is padding_mask gives presoftmax weights
    #             ignore_mask = torch.stack([context == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1)
    #             context = context.masked_fill_(ignore_mask, self.padding_idx)  # this line and the previous one removes the special characters
    #             prevs, nexts = lm_logits[:, :-1, :].contiguous(), context[:, 1:].contiguous()
    #             batch_lm_loss += (self.lm_criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1)) / len(contexts))
    #
    #     prevs, nexts = targets[:, :-1].contiguous(), targets[:, 1:].contiguous()
    #     outputs = self.decode(prevs, enc_contexts)  # this gives the pre-softmax
    #     outputs = F.log_softmax(outputs, dim=-1)
    #     batch_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))
    #
    #     return batch_lm_loss, batch_loss

    #todo the following are the previous way of implementing the beam search

    # def predict(self, contexts=[]):
    #     enc_contexts = [self.encode(c) for c in contexts]
    #     prediction = self.beam_search(enc_contexts)
    #
    #     return prediction


    # def beam_search(self, enc_contexts=[], return_beams=False):
    #     with torch.no_grad():
    #         if len(enc_contexts) == 0:
    #             return []
    #
    #         batch_size = enc_contexts[0][0].shape[0]
    #         device = next(self.parameters()).device
    #
    #         prevs = torch.full((batch_size * self.beam_size, 1), fill_value=self.bos_id, dtype=torch.long, device=device)
    #
    #         beam_scores = torch.zeros(batch_size, self.beam_size, device=device)
    #         beam_lens = torch.ones(batch_size, self.beam_size, dtype=torch.long, device=device)
    #         is_end = torch.zeros(batch_size, self.beam_size, dtype=torch.uint8, device=device)
    #
    #         beam_enc_contexts = []
    #         for c, p in enc_contexts:
    #             c = c.unsqueeze(1).repeat(1, self.beam_size, 1, 1)
    #             c = c.view(-1, c.shape[2], c.shape[3])
    #             p = p.unsqueeze(1).repeat(1, self.beam_size, 1)
    #             p = p.view(-1, p.shape[2])
    #             beam_enc_contexts.append((c, p))
    #
    #         current_sample_prob = 1
    #         group_size = self.beam_size // self.diversity_groups
    #         diversity_penalty = torch.zeros((batch_size, self.n_embeddings), device=device)
    #
    #         for i in range(self.max_seq_len):
    #             outputs, _ = self.transformer_module(prevs, beam_enc_contexts)
    #
    #             logits = self.generate(outputs[:, -1, :])
    #             log_probs = F.log_softmax(logits, dim=-1)
    #             log_probs = log_probs.view(batch_size, self.beam_size, -1)
    #
    #             beam_scores = beam_scores.unsqueeze(-1) + log_probs * (1 - is_end.float().unsqueeze(-1))
    #             penalty = self._length_penalty(beam_lens.float() + 1 - is_end.float())
    #             penalty = penalty.unsqueeze(-1).repeat(1, 1, self.n_embeddings)
    #             beam_scores = beam_scores / penalty
    #
    #             if i == 0:
    #                 penalty = penalty[:, 0, :]
    #                 beam_scores = beam_scores[:, 0, :]
    #
    #                 beam_scores, idxs = beam_scores.topk(self.beam_size, dim=-1)
    #                 beam_idxs = torch.zeros((batch_size, self.beam_size), dtype=torch.long, device=device)
    #             else:
    #                 penalty = penalty.view(batch_size, self.diversity_groups, group_size, -1)
    #                 beam_scores = beam_scores.view(batch_size, self.diversity_groups, group_size, -1)
    #
    #                 all_scores, all_idxs = [], []
    #                 for g in range(self.diversity_groups):
    #                     g_beam_scores = beam_scores[:, g, :, :]
    #                     g_penalty = penalty[:, g, :, :]
    #                     g_beam_scores -= self.diversity_coef * diversity_penalty.unsqueeze(1) / g_penalty
    #                     g_beam_scores = g_beam_scores.view(batch_size, -1)
    #
    #                     if random.random() < current_sample_prob:
    #                         beam_probas = F.softmax(g_beam_scores, dim=-1)
    #                         if self.annealing_topk is not None:
    #                             beam_probas, sample_idxs = beam_probas.topk(self.annealing_topk, dim=-1)
    #                             g_idxs = torch.multinomial(beam_probas, group_size)
    #                             g_idxs = torch.gather(sample_idxs, 1, g_idxs)
    #                         else:
    #                             g_idxs = torch.multinomial(beam_probas, group_size)
    #                     else:
    #                         _, g_idxs = g_beam_scores.topk(group_size, dim=-1)
    #
    #                     g_scores = torch.gather(beam_scores[:, g, :, :].view(batch_size, -1), 1, g_idxs)
    #                     g_idxs += g * group_size * self.n_embeddings
    #
    #                     all_scores.append(g_scores)
    #                     all_idxs.append(g_idxs)
    #
    #                     diversity_penalty.scatter_add_(1, torch.fmod(g_idxs, self.n_embeddings), torch.ones((batch_size, group_size), device=device))
    #
    #                 diversity_penalty.fill_(0)
    #                 penalty = penalty.view(batch_size, -1)
    #                 beam_scores = torch.cat(all_scores, dim=-1)
    #                 idxs = torch.cat(all_idxs, dim=-1)
    #
    #                 beam_idxs = (idxs.float() / self.n_embeddings).long()
    #
    #             penalty = torch.gather(penalty, 1, idxs)
    #             sym_idxs = torch.fmod(idxs, log_probs.shape[-1])
    #             is_end = torch.gather(is_end, 1, beam_idxs)
    #             beam_lens = torch.gather(beam_lens, 1, beam_idxs)
    #
    #             sym_idxs[is_end] = self.padding_idx
    #             beam_lens[~is_end] += 1
    #             is_end[sym_idxs == self.eos_id] = 1
    #
    #             sym_idxs = sym_idxs.view(batch_size * self.beam_size, 1)
    #             prevs = prevs.view(batch_size, self.beam_size, -1)
    #             prevs = torch.gather(prevs, 1, beam_idxs.unsqueeze(-1).repeat(1, 1, prevs.shape[-1]))
    #             prevs = prevs.view(batch_size * self.beam_size, -1)
    #             prevs = torch.cat([prevs, sym_idxs], dim=1)
    #
    #             if all(is_end.view(-1)):
    #                 break
    #
    #             beam_scores *= penalty
    #             current_sample_prob *= self.annealing
    #
    #         predicts = []
    #         result = prevs.view(batch_size, self.beam_size, -1)
    #
    #         if return_beams:
    #              return result, beam_lens
    #
    #         if self.sample:
    #             probs = F.softmax(beam_scores, dim=-1)
    #             bests = torch.multinomial(probs, 1).view(-1)
    #         else:
    #             bests = beam_scores.argmax(dim=-1)
    #
    #         for i in range(batch_size):
    #             best_len = beam_lens[i, bests[i]]
    #             best_seq = result[i, bests[i], 1:best_len-1]
    #             predicts.append(best_seq.tolist())
    #
    #     return predicts
