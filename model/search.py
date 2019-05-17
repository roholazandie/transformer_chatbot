import torch
import torch.nn.functional as F
import random


class SamplingSearch():

    def __init__(self, model, max_seq_len, temperature=1.0, top_k=0, top_p=0.9, special_tokens_ids = [],no_sample=True):
        # todo: sampling can be augmented with beam search strategy
        # todo: below should go inside a loop
        self.model = model
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.special_tokens_ids = special_tokens_ids
        self.no_sample = no_sample
        self.min_length = 1

    def search(self, enc_contexts=[]):

        def top_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
            """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
                Args:
                    logits: logits distribution shape (..., vocabulary size)
                    top_k >0: keep only top k tokens with highest probability (top-k filtering).
                    top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            """
            top_k = min(top_k, logits.size(-1))  # Safety check
            if top_k > 0:
                # Remove all tokens with a probability less than the last token of the top-k
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = filter_value

            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = filter_value
            return logits

        batch_size = enc_contexts[0][0].shape[0]
        device = next(self.model.parameters()).device
        prevs = torch.full((batch_size, 1), fill_value=self.model.bos_id, dtype=torch.long, device=device)


        current_output = []
        for i in range(self.max_seq_len):
            # Get logits with a forward pass in our model (input is pre-defined)
            outputs, _ = self.model.transformer_module(prevs, enc_contexts)
            logits = self.model.generate(outputs[:, -1, :])
            # Keep only the last token predictions, apply a temperature coefficient and filter
            logits = logits[0, :] / self.temperature
            logits = top_filtering(logits, top_k=self.top_k, top_p=self.top_p)
            probs = F.softmax(logits, dim=-1)

            # Sample from the filtered distribution
            prev = torch.topk(probs, 1)[1] if self.no_sample else torch.multinomial(probs, 1)
            # if i < self.min_length and prev.item() in special_tokens_ids:  # todo rooh: to remove special tokens
            #     while prev.item() in special_tokens_ids:
            #         prev = torch.multinomial(probs, num_samples=1)
            #
            if prev.item() in self.special_tokens_ids:
                break
            current_output.append(prev.item())
            prevs = torch.cat([prevs, prev.unsqueeze(0)], dim=1)



        return [current_output]


class BeamSearch():

    def __init__(self, model,
                 beam_size=5,
                 diversity_groups=1,
                 length_penalty_coef=0.8,
                 max_seq_len=256,
                 diversity_coef=0,
                 sample=False,
                 annealing=0,
                 annealing_topk=None):
        self.model = model
        self.beam_size = beam_size
        self.diversity_groups = diversity_groups
        self.n_embeddings = model.n_embeddings
        self.length_penalty_coef = length_penalty_coef
        self.diversity_coef = diversity_coef
        self.max_seq_len = max_seq_len
        self.sample = sample
        self.annealing = annealing
        self.annealing_topk = annealing_topk

    def _length_penalty(self, sequence_lengths):
        """https://arxiv.org/abs/1609.08144"""
        return (5 + sequence_lengths) ** self.length_penalty_coef / (5 + 1) ** self.length_penalty_coef

    def predict(self, contexts=[]):  # todo probably we need a class generator to have predict
        enc_contexts = [self.model.encode(c) for c in contexts]
        prediction = self.search(enc_contexts)

        return prediction

    def search(self, enc_contexts=[], return_beams=False): #by default no beams will be returned
        with torch.no_grad():
            if len(enc_contexts) == 0:
                return []

            batch_size = enc_contexts[0][0].shape[0]
            device = next(self.model.parameters()).device

            prevs = torch.full((batch_size * self.beam_size, 1), fill_value=self.model.bos_id, dtype=torch.long,
                               device=device)

            beam_scores = torch.zeros(batch_size, self.beam_size, device=device)
            beam_lens = torch.ones(batch_size, self.beam_size, dtype=torch.long, device=device)
            is_end = torch.zeros(batch_size, self.beam_size, dtype=torch.uint8, device=device)

            beam_enc_contexts = []
            for c, p in enc_contexts:
                c = c.unsqueeze(1).repeat(1, self.beam_size, 1, 1)
                c = c.view(-1, c.shape[2], c.shape[3])
                p = p.unsqueeze(1).repeat(1, self.beam_size, 1)
                p = p.view(-1, p.shape[2])
                beam_enc_contexts.append((c, p))

            current_sample_prob = 1
            group_size = self.beam_size // self.diversity_groups
            diversity_penalty = torch.zeros((batch_size, self.n_embeddings), device=device)

            for i in range(self.max_seq_len):
                outputs, _ = self.model.transformer_module(prevs, beam_enc_contexts)

                logits = self.model.generate(outputs[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1)
                log_probs = log_probs.view(batch_size, self.beam_size, -1)

                beam_scores = beam_scores.unsqueeze(-1) + log_probs * (1 - is_end.float().unsqueeze(-1))
                penalty = self._length_penalty(beam_lens.float() + 1 - is_end.float())
                penalty = penalty.unsqueeze(-1).repeat(1, 1, self.n_embeddings)
                beam_scores = beam_scores / penalty

                if i == 0:
                    penalty = penalty[:, 0, :]
                    beam_scores = beam_scores[:, 0, :]

                    beam_scores, idxs = beam_scores.topk(self.beam_size, dim=-1)
                    beam_idxs = torch.zeros((batch_size, self.beam_size), dtype=torch.long, device=device)
                else:
                    penalty = penalty.view(batch_size, self.diversity_groups, group_size, -1)
                    beam_scores = beam_scores.view(batch_size, self.diversity_groups, group_size, -1)

                    all_scores, all_idxs = [], []
                    for g in range(self.diversity_groups):
                        g_beam_scores = beam_scores[:, g, :, :]
                        g_penalty = penalty[:, g, :, :]
                        g_beam_scores -= self.diversity_coef * diversity_penalty.unsqueeze(1) / g_penalty
                        g_beam_scores = g_beam_scores.view(batch_size, -1)

                        if random.random() < current_sample_prob:
                            beam_probas = F.softmax(g_beam_scores, dim=-1)
                            if self.annealing_topk is not None:
                                beam_probas, sample_idxs = beam_probas.topk(self.annealing_topk, dim=-1)
                                g_idxs = torch.multinomial(beam_probas, group_size)
                                g_idxs = torch.gather(sample_idxs, 1, g_idxs)
                            else:
                                g_idxs = torch.multinomial(beam_probas, group_size)
                        else:
                            _, g_idxs = g_beam_scores.topk(group_size, dim=-1)

                        g_scores = torch.gather(beam_scores[:, g, :, :].view(batch_size, -1), 1, g_idxs)
                        g_idxs += g * group_size * self.n_embeddings

                        all_scores.append(g_scores)
                        all_idxs.append(g_idxs)

                        diversity_penalty.scatter_add_(1, torch.fmod(g_idxs, self.n_embeddings),
                                                       torch.ones((batch_size, group_size), device=device))

                    diversity_penalty.fill_(0)
                    penalty = penalty.view(batch_size, -1)
                    beam_scores = torch.cat(all_scores, dim=-1)
                    idxs = torch.cat(all_idxs, dim=-1)

                    beam_idxs = (idxs.float() / self.n_embeddings).long()

                penalty = torch.gather(penalty, 1, idxs)
                sym_idxs = torch.fmod(idxs, log_probs.shape[-1])#element-wise remainder of division
                is_end = torch.gather(is_end, 1, beam_idxs)
                beam_lens = torch.gather(beam_lens, 1, beam_idxs)

                sym_idxs[is_end] = self.model.padding_idx
                beam_lens[~is_end] += 1
                is_end[sym_idxs == self.model.eos_id] = 1

                sym_idxs = sym_idxs.view(batch_size * self.beam_size, 1)
                prevs = prevs.view(batch_size, self.beam_size, -1)
                prevs = torch.gather(prevs, 1, beam_idxs.unsqueeze(-1).repeat(1, 1, prevs.shape[-1]))
                prevs = prevs.view(batch_size * self.beam_size, -1)
                prevs = torch.cat([prevs, sym_idxs], dim=1)

                if all(is_end.view(-1)):
                    break

                beam_scores *= penalty
                current_sample_prob *= self.annealing

            predicts = []
            result = prevs.view(batch_size, self.beam_size, -1)  # B x b x T

            if return_beams:
                return result, beam_lens

            if self.sample:
                probs = F.softmax(beam_scores, dim=-1)
                bests = torch.multinomial(probs, 1).view(-1)
            else:
                bests = beam_scores.argmax(dim=-1)

            for i in range(batch_size):
                best_len = beam_lens[i, bests[i]]
                best_seq = result[i, bests[i], 1:best_len - 1]
                predicts.append(best_seq.tolist())

        return predicts
