import torch
import torch.nn.functional as F
from collections import deque
from collections import defaultdict

from model.search import BeamSearch, SamplingSearch
from parlai.core.agents import Agent
from model.transformer_model import TransformerModel
from utils.text import BPEVocab
from utils.common import pad_sequence
from model.postprocessing import ngram_replaser, ReplyChecker, detokenize, syntax_fix
from utils.retrieval import RetrievalBot, DIALOG_SIZE
from utils.sentiment import pick_emoji, clean_emoji
from model.config import get_model_config
from projects.convai2.build_dict import build_dict
import random


class TransformerAgent(Agent):
    @staticmethod
    def add_cmdline_args(argparser):
        agent_args = argparser.add_argument_group('Agent parameters')
        agent_args.add_argument('-gpu', '--gpu', type=int, default=-1,
                                help='which GPU to use')
        agent_args.add_argument('--no-cuda', type=bool, default=False,
                                help='disable GPUs even if available. otherwise, will use GPUs if '
                                     'available on the device.')
        agent_args.add_argument('--rank_candidates', type=bool, default=False,
                                help='Whether the model should parse candidates for ranking.')
        agent_args.add_argument('--sample', type=bool, default=False,
                                help='Sampling of beam from beam search')
        agent_args.add_argument('--wild_mode', type=bool, default=False,
                                help='')
        agent_args.add_argument('--replace_repeat', type=bool, default=True,
                                help='')
        agent_args.add_argument('--replace_ngram', type=bool, default=True,
                                help='')
        agent_args.add_argument('--detokenize', type=bool, default=True,
                                help='')
        agent_args.add_argument('--emoji_prob', type=float, default=0.5,
                                help='')
        agent_args.add_argument('--ngram_size', type=int, default=3,
                                help='')
        agent_args.add_argument('--add_questions', type=float, default=0.3,
                                help='')
        agent_args.add_argument('--clean_emoji', type=bool, default=True,
                                help='')
        agent_args.add_argument('--check_grammar', type=bool, default=True,
                                help='')
        agent_args.add_argument('--correct_generative', type=bool, default=True,
                                help='')
        agent_args.add_argument('--split_into_sentences', type=bool, default=True,
                                help='')

        agent_args.add_argument('--max_seq_len', type=int, default=128,
                                help='')
        agent_args.add_argument('--beam_size', type=int, default=1,
                                help='')
        agent_args.add_argument('--diversity_coef', type=float, default=0,
                                help='')
        agent_args.add_argument('--diversity_groups', type=int, default=1,
                                help='')
        agent_args.add_argument('--annealing_topk', type=float, default=None,
                                help='')
        agent_args.add_argument('--annealing', type=float, default=0.0,
                                help='')
        agent_args.add_argument('--length_penalty', type=float, default=0.6,
                                help='')

        return argparser

    def __init__(self, opt, shared=None):
        super(TransformerAgent, self).__init__(opt, shared)

        self.use_cuda = not self.opt.get('no_cuda') and torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.set_device(self.opt['gpu'])

        torch.set_grad_enabled(False)

        model_config = get_model_config()
        self.vocab = BPEVocab.from_files(model_config.bpe_vocab_path, model_config.bpe_codes_path)
        self.reply_checker = ReplyChecker(correct_generative=self.opt['correct_generative'],
                                          split_into_sentences=self.opt['split_into_sentences'])

        self.replace_repeat = self.opt['replace_repeat']
        self.replace_ngram = self.opt['replace_ngram']
        self.ngram_size = self.opt['ngram_size']
        self.detokenize = self.opt['detokenize']
        self.emoji_prob = self.opt['emoji_prob']
        self.add_questions = self.opt['add_questions']
        self.beam_size = self.opt['beam_size']

        self.clean_emoji = self.opt['clean_emoji']
        self.check_grammar = self.opt['check_grammar']

        # 'max_seq_len': 128,
        # 'beam_size': 1,
        # 'diversity_coef': 0,
        # 'diversity_groups': 1,
        # 'annealing_topk': None,
        # 'annealing': 0,
        # 'length_penalty': 0.6,

        convai_dict = build_dict()
        self.prefix2words = self.get_prefix2words(convai_dict)

        if self.opt['annealing_topk'] is not None:
            assert self.opt['annealing_topk'] > self.opt['beam_size']

        assert self.opt['diversity_coef'] >= 0
        assert self.opt['beam_size'] % self.opt['diversity_groups'] == 0

        if shared is None:
            self.model = TransformerModel(n_layers=model_config.n_layers,
                                          n_embeddings=len(self.vocab),
                                          n_pos_embeddings=model_config.n_pos_embeddings,
                                          embeddings_size=model_config.embeddings_size,
                                          padding_idx=self.vocab.pad_id,
                                          n_heads=model_config.n_heads,
                                          dropout=model_config.dropout,
                                          embed_dropout=model_config.embed_dropout,
                                          attn_dropout=model_config.attn_dropout,
                                          ff_dropout=model_config.ff_dropout,
                                          bos_id=self.vocab.bos_id,
                                          eos_id=self.vocab.eos_id,
                                          n_segments=model_config.n_segments,
                                          )

            if model_config.searcher == "beam":
                self.searcher = BeamSearch(self.model,
                                           beam_size=model_config.beam_size,
                                           diversity_groups=model_config.diversity_groups,
                                           length_penalty_coef=model_config.length_penalty,
                                           max_seq_len=model_config.max_seq_len,
                                           diversity_coef=model_config.diversity_coef,
                                           sample=False,
                                           annealing=model_config.annealing,
                                           annealing_topk=model_config.annealing_topk
                                           )

            elif model_config.searcher == "sampling":
                self.searcher = SamplingSearch(self.model,
                                               max_seq_len=model_config.max_seq_len,
                                               temperature=0.7,
                                               top_k=0,
                                               top_p=0.9,
                                               special_tokens_ids=self.vocab.special_tokens_ids)
            else:
                raise ValueError("Must specify the searcher in config")

            self.retrieval_bot = RetrievalBot()

            pretrained_state_dict = torch.load(model_config.checkpoint_path, map_location=lambda storage, loc: storage)

            # todo this part of the code is temporary
            # in the pretrained model the name is pre_softmax.weight
            if 'pre_softmax.weight' in pretrained_state_dict['model']:
                pretrained_state_dict['model']['decoder.weight'] = pretrained_state_dict['model']['pre_softmax.weight']
                del pretrained_state_dict['model']['pre_softmax.weight']

            if 'model' in pretrained_state_dict:
                pretrained_state_dict = pretrained_state_dict['model']

            self.model.load_state_dict(pretrained_state_dict)
            print('Weights loaded from {}'.format(model_config.checkpoint_path))

            if self.use_cuda:
                self.model = self.model.cuda()

            self.model.eval()

        else:
            self.model = shared['model']
            self.retrieval_bot = shared['retrieval']

        self.reset()

    def _preprocess_text(self, text):
        if self.clean_emoji:
            text = clean_emoji(text)

        if self.check_grammar:
            text = syntax_fix(text).lower()

        return text

    def _parse(self, text):
        # todo: fix grammar mistakes?
        persona_info = []
        dialog = []
        for subtext in text.split('\n'):
            subtext = subtext.strip()

            if self.opt['wild_mode'] and len(self.history['info']) == 0 and len(self.history['dialog']) == 0:
                subtext = 'your persona: ' + subtext

            if subtext.startswith('your persona:'):
                subtext = subtext.replace('your persona:', '').strip()
                subtext = self._preprocess_text(subtext).strip()
                persona_info.append(subtext)
            else:
                subtext = self._preprocess_text(subtext).strip()
                dialog.append(subtext)

        return persona_info, dialog

    def observe(self, observation):
        if self.episode_done:
            self.reset()

        if 'text' in observation:
            text = observation['text']
            persona_info, dialog = self._parse(text)

            if persona_info:
                self.history['str_info'] = ' '.join(persona_info)
            self.history['str_dialog'].extend(dialog)

            persona_info = sum([self.vocab.string2ids(i) for i in persona_info], [])  # concatenate all ids
            self.history['info'].extend(persona_info)

            for i, d in enumerate(dialog, 1):
                d = self.vocab.string2ids(d)
                if i % 2 == 1:
                    d = [self.vocab.talker1_bos_id] + d + [self.vocab.talker1_eos_id]
                else:
                    d = [self.vocab.talker2_bos_id] + d + [self.vocab.talker2_eos_id]

                self.history['dialog'].extend(d)

        observation['agent'] = self  # what's the use of this?

        self.episode_done = observation['episode_done']
        self.observation = observation

        return observation

    def act(self):
        return self.batch_act([self.observation])[0]  # select the first one of the batch (the batch size is one so we select the first one)

    def _postprocess_text(self, reply, agent):
        str_reply = self.vocab.ids2string(reply)

        if self.replace_repeat:
            str_reply = agent.reply_checker.check_reply(str_reply,
                                                        agent.history['str_dialog'][-1],
                                                        agent.history['str_info'])

        if self.beam_size > 1 and random.uniform(0, 1) < self.add_questions and '?' not in str_reply:
            question = self.retrieval_bot.generate_question(list(agent.history['str_dialog']),
                                                            agent.history['str_info'])
            if question is not None and question not in str_reply:
                str_reply = ' '.join([str_reply, question])

        if self.replace_ngram:
            str_reply = ngram_replaser(agent.history['str_info'], str_reply, n=self.ngram_size)

        reply = self.vocab.string2ids(str_reply)

        if self.detokenize:
            str_reply = detokenize(str_reply)

        if random.uniform(0, 1) < self.emoji_prob:
            str_reply = ' '.join([str_reply, pick_emoji(str_reply)])

        return str_reply, reply

    def batch_act(self, observations):
        def is_valid_history(history):
            return len(history['dialog'])

        def to_tensor(string):
            ids = [self.vocab.bos_id] + self.vocab.string2ids(string) + [self.vocab.eos_id]
            return torch.tensor(ids, dtype=torch.long)

        batch_reply = [{'id': self.getID(), 'text': '', 'text_candidates': []} for _ in range(len(observations))]
        valid_ids = [i for i, obs in enumerate(observations) if is_valid_history(obs['agent'].history)]
        batch_size = len(valid_ids)

        if batch_size == 0:
            return batch_reply

        try:
            valid_observations = [observations[i] for i in valid_ids]
            # we trim to model.n_pos_embeddings-3 to be able to add the eos and bos and fit to network
            print(observations[0]['agent'].history['str_info'])
            infos = [obs['agent'].history['info'][:self.model.n_pos_embeddings - 3] for obs in valid_observations]
            infos = [([self.vocab.info_bos_id] + ifo + [self.vocab.info_eos_id] if len(ifo) else ifo) for ifo in infos]
            dialogs = [list(obs['agent'].history['dialog'])[-self.model.n_pos_embeddings + 1:] for obs in
                       valid_observations]
            contexts = []

            if max(map(len, infos)) > 0:
                infos = [torch.tensor(i, dtype=torch.long) for i in infos]
                infos = pad_sequence(infos, batch_first=True, padding_value=self.model.padding_idx)
                if self.use_cuda:
                    infos = infos.cuda()
                contexts.append(infos)

            if max(map(len, dialogs)) > 0:
                dialogs = [torch.tensor(d, dtype=torch.long) for d in dialogs]
                dialogs = pad_sequence(dialogs, batch_first=True, padding_value=self.model.padding_idx)
                if self.use_cuda:
                    dialogs = dialogs.cuda()
                contexts.append(dialogs)

            # todo the following two lines can be replaced with self.searcher.predict(contexts)
            enc_contexts = [self.model.encode(c) for c in contexts]
            pred_texts = self.searcher.search(enc_contexts)

            for i in range(batch_size):
                # this is with retrieval bot
                #pred_text_str, pred_text = self._postprocess_text(pred_texts[i], valid_observations[i]['agent'])

                # wihtout retrieval bot
                pred_text_str = self.vocab.ids2string(pred_texts[i])
                pred_text = pred_texts[i]

                valid_observations[i]['agent'].history['dialog'].extend([self.vocab.talker2_bos_id] +
                                                                        pred_text +
                                                                        [self.vocab.talker2_eos_id])
                batch_reply[valid_ids[i]]['text'] = pred_text_str
                batch_reply[valid_ids[i]]['episode_done'] = valid_observations[i]['agent'].episode_done

            if self.opt['rank_candidates']:
                candidates = [list(obs.get('label_candidates', [])) for obs in valid_observations]

                # for i in range(len(candidates)):
                #     candidates[i].append(observations[i]['agent'].history['str_info'].split('.')[0])

                lens_candidates = [len(c) for c in candidates]

                if max(lens_candidates) > 0:
                    candidates = [c + ['' for _ in range(max(lens_candidates) - len(c))] for c in candidates]
                    scores = [[] for _ in range(len(candidates))]

                    for i in range(max(lens_candidates)):
                        current_cands = [to_tensor(c[i])[:self.model.n_pos_embeddings - 1] for c in candidates]
                        current_cands = pad_sequence(current_cands, batch_first=True,
                                                     padding_value=self.model.padding_idx)
                        if self.use_cuda:
                            current_cands = current_cands.cuda()

                        logits = self.model.decode(current_cands[:, :-1], enc_contexts)
                        log_probas = F.log_softmax(logits, dim=-1)
                        log_probas = torch.gather(log_probas, -1, current_cands[:, 1:].unsqueeze(-1)).squeeze(-1)
                        log_probas.masked_fill_(current_cands[:, 1:].eq(self.model.padding_idx), 0)

                        current_lens = current_cands[:, 1:].ne(self.model.padding_idx).float().sum(dim=-1)
                        current_scores = log_probas.sum(dim=-1) / current_lens #longer resposes corresponds to lower score

                        for k, s in enumerate(current_scores):
                            if i < lens_candidates[k]:
                                scores[k].append(s.item())


                        # my_candidate = to_tensor(observations[i]['agent'].history['info'])[:self.model.n_pos_embeddings - 1]
                        # my_candidate = pad_sequence(my_candidate.unsqueeze(0), batch_first=True, padding_value=self.model.padding_idx)
                        #
                        # my_candidate = my_candidate.cuda()
                        # logit = self.model.decode(my_candidate[:, :-1], enc_contexts)
                        # log_prob = F.log_softmax(logit, dim=-1)
                        # log_prob = torch.gather(log_prob, -1, my_candidate[:, 1:].unsqueeze(-1)).squeeze(-1)
                        # log_prob.masked_fill_(current_cands[:, 1:].eq(self.model.padding_idx), 0)
                        #
                        # my_scores = log_prob.sum(dim=-1) / len(my_candidate)



                    ranked_ids = [sorted(range(len(s)), key=lambda k: s[k], reverse=True) for s in scores]
                    ranked_strings = [[c[i] for i in ids] for ids, c in zip(ranked_ids, candidates)]

                    for i in range(batch_size):
                        batch_reply[valid_ids[i]]['text_candidates'] = ranked_strings[i]

                        # conversation = observations[i]['text']
                        # choosed = ranked_strings[i][0]
                        # label = observations[i]['eval_labels']
                        # print("conversation ", conversation)
                        # print("choosed: "+str(i), choosed)
                        # print("label "+str(i), label)
        except Exception as e:
            # raise e
            print(e)

        return batch_reply


    def next_word_probability(self, partial_out):
        def is_valid_history(history):
            return len(history['dialog'])

        def to_tensor(string):
            ids = [self.vocab.bos_id] + self.vocab.string2ids(string) + [self.vocab.eos_id]
            return torch.tensor(ids, dtype=torch.long)

        if is_valid_history(self.observation['agent'].history):
            persona_info = self.observation['agent'].history['info'][:self.model.n_pos_embeddings - 3]
            persona_info = [self.vocab.info_bos_id] + persona_info + [self.vocab.info_eos_id]

            dialog = list(self.observation['agent'].history['dialog'])[-self.model.n_pos_embeddings+1:]

            context = []
            if len(persona_info) > 0:
                info = torch.tensor(persona_info, dtype=torch.long)
                info = pad_sequence(info.unsqueeze(0), batch_first=True, padding_value=self.model.padding_idx)
                if self.use_cuda:
                    info = info.cuda()
                context.append(info)

            if len(dialog) > 0:
                dialog = torch.tensor(dialog, dtype=torch.long)
                dialog = pad_sequence(dialog.unsqueeze(0), batch_first=True, padding_value=self.model.padding_idx)
                if self.use_cuda:
                    dialog = dialog.cuda()
                context.append(dialog)

            enc_contexts = [self.model.encode(c) for c in context]

            # pred_texts = self.searcher.search(enc_contexts)
            # str_reply = self.vocab.ids2string(pred_texts[0])
            # print(str_reply)


            partial_out_id = to_tensor(' '.join(partial_out))[:self.model.n_pos_embeddings - 1]
            partial_out_id = pad_sequence(partial_out_id.unsqueeze(0), batch_first=True, padding_value=self.model.padding_idx)
            if self.use_cuda:
                partial_out_id = partial_out_id.cuda()

            with torch.no_grad():
                outputs, _ = self.model.transformer_module(partial_out_id, enc_contexts)

            logits = self.model.generate(outputs[:, -1, :]).squeeze(0)
            probs = F.softmax(logits)

            print(self.prefix2words[probs.argmax().item()], self.vocab.id2token[probs.argmax().item()])
            print()
            #probs = probs[torch.randperm(len(probs))]

            dist = {}
            for prefix_id, words in self.prefix2words.items():
                for word, ratio in words.items():
                    dist[word] = probs[prefix_id].item() * ratio

            # values = list(dist.values())
            # keys = list(dist.keys())
            # random.shuffle(values)
            # dist1 = {}
            # for key, value in zip(keys, values):
            #     dist1[key] = value
            return dist

    def share(self):
        shared = super(TransformerAgent, self).share()
        shared['opt'] = self.opt
        shared['model'] = self.model
        shared['retrieval'] = self.retrieval_bot

        return shared

    def reset(self):
        self.history = {'str_info': None, 'str_dialog': deque(DIALOG_SIZE * ['None'], maxlen=DIALOG_SIZE),
                        'info': [], 'dialog': deque(maxlen=self.model.n_pos_embeddings - 1)}
        self.episode_done = True
        self.observation = None
        self.reply_checker.clean()


    def get_prefix2words(self, convai_dict, smoothing_freq=5):
        """ map BPE-prefix => dict(full_words beginning with BPE-prefix, associated words_counts) """
        prefix2words = defaultdict(dict)
        for i in range(len(convai_dict)):
            word = convai_dict[i]
            freq = convai_dict.freq[word] + smoothing_freq
            bpe_tokens = self.vocab._bpe(word)
            prefix_id = self.vocab.token2id[bpe_tokens[0]]
            prefix2words[prefix_id].update(dict([(word, freq)]))

        for prefix_id, words in prefix2words.items():
            total_counts = sum(words.values())
            prefix2words[prefix_id] = dict((word, count/total_counts) for word, count in words.items())

        return prefix2words