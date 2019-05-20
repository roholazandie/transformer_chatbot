import torch
from torch import nn
import random

from model.loss import LabelSmoothingLoss
from utils.common import load_openai_weights, set_seed
from utils.metrics import f1_risk, f1_score, bleu_score
from model.transformer_model import TransformerModel
from model.trainer import Trainer
from utils.text import BPEVocab
from model.dataset import FacebookDataset
from model.config import get_model_config, get_trainer_config
from torch.utils.data import DataLoader
from utils.common import pad_sequence
from model.optim import Adam, NoamOpt
from model.search import BeamSearch
import missinglink




def get_train_test_loaders(train_dataset, test_dataset, batch_size, batch_split, n_jobs, collate_func):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size // batch_split, shuffle=True,
                                  num_workers=n_jobs, collate_fn=collate_func)
    test_dataloader = None
    if test_dataset is not None:
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size // batch_split, shuffle=False,
                                     num_workers=n_jobs, collate_fn=collate_func)

    return train_dataloader, test_dataloader


def main():
    missinglink_project = missinglink.PyTorchProject()

    model_config = get_model_config()
    trainer_config = get_trainer_config()

    set_seed(trainer_config.seed)
    device = torch.device(trainer_config.device)

    vocab = BPEVocab.from_files(model_config.bpe_vocab_path, model_config.bpe_codes_path)

    # todo need to be added if we want to use TransformerLMHead
    # lm_criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id).to(device)
    # criterion = LabelSmoothingLoss(n_labels=len(vocab),
    #                                smoothing=trainer_config.label_smoothing,
    #                                ignore_index=vocab.pad_id).to(device)


    transformer = TransformerModel(n_layers=model_config.n_layers,
                                   n_embeddings=len(vocab),
                                   n_pos_embeddings=model_config.n_pos_embeddings,
                                   embeddings_size=model_config.embeddings_size,
                                   padding_idx=vocab.pad_id,
                                   n_heads=model_config.n_heads,
                                   dropout=model_config.dropout,
                                   embed_dropout=model_config.embed_dropout,
                                   attn_dropout=model_config.attn_dropout,
                                   ff_dropout=model_config.ff_dropout,
                                   bos_id=vocab.bos_id,
                                   eos_id=vocab.eos_id,
                                   max_seq_len=model_config.max_seq_len,
                                   n_segments=model_config.n_segments,
                                   )

    searcher = BeamSearch(transformer,
                          beam_size=model_config.beam_size,
                          diversity_groups=model_config.diversity_groups,
                          length_penalty_coef=model_config.length_penalty,
                          max_seq_len=model_config.max_seq_len,
                          diversity_coef=model_config.diversity_coef,
                          sample=False,
                          annealing=model_config.annealing,
                          annealing_topk=model_config.annealing_topk
                          )


    if not trainer_config.load_last:
        load_openai_weights(transformer.transformer_module,
                            trainer_config.openai_parameters_dir,
                            n_special_tokens=vocab.n_special_tokens)
        print('OpenAI weights loaded from {}'.format(trainer_config.openai_parameters_dir))

    train_dataset = FacebookDataset(trainer_config.train_datasets, vocab, transformer.n_pos_embeddings - 1)
    test_dataset = FacebookDataset(trainer_config.test_datasets, vocab, transformer.n_pos_embeddings - 1)

    train_loader, test_loader = get_train_test_loaders(train_dataset, test_dataset,
                                                       batch_size=trainer_config.batch_size,
                                                       batch_split=trainer_config.batch_split,
                                                       n_jobs=trainer_config.n_jobs,
                                                       collate_func=train_dataset.collate_func)

    base_optimizer = Adam(transformer.parameters(), lr=trainer_config.lr, weight_decay=0.01)
    optimizer = NoamOpt(transformer.embeddings_size, trainer_config.lr, trainer_config.lr_warmup, base_optimizer)

    model_trainer = Trainer(transformer,
                            searcher,
                            train_loader,
                            test_loader,
                            optimizer=optimizer,
                            batch_size=trainer_config.batch_size,
                            batch_split=trainer_config.batch_split,
                            lr=trainer_config.lr,
                            lr_warmup=trainer_config.lr_warmup,
                            lm_weight=trainer_config.lm_weight,
                            risk_weight=trainer_config.risk_weight,
                            test_period=trainer_config.test_period,
                            clip_grad=trainer_config.clip_grad,
                            last_checkpoint_path=trainer_config.last_checkpoint_path,
                            device=device,
                            vocab=vocab)

    if trainer_config.load_last:
        model_dict = model_trainer.state_dict()
        pretrained_state_dict = torch.load(trainer_config.last_checkpoint_path, map_location=device)
        #pretrained_state_dict = {k: v for k, v in pretrained_state_dict['model'].items() if k in model_dict['model']}
        pretrained_state_dict['model']['decoder.weight'] = pretrained_state_dict['model']['pre_softmax.weight']
        del pretrained_state_dict['model']['pre_softmax.weight']
        model_trainer.load_state_dict(pretrained_state_dict)
        print('Weights loaded from {}'.format(trainer_config.last_checkpoint_path))

    try:
        with missinglink_project.create_experiment(model=transformer,
                                                   optimizer=optimizer,
                                                   train_data_object=train_loader) as experiment:
            after_epoch_func = []  # [save_func, sample_text_func, test_func]
            model_trainer.train(trainer_config.n_epochs,
                                after_epoch_funcs=after_epoch_func,
                                risk_func=f1_risk,
                                experiment=experiment)



    except (KeyboardInterrupt, Exception, RuntimeError) as e:
        torch.save(model_trainer.state_dict(), trainer_config.interrupt_checkpoint_path)
        raise e


if __name__ == '__main__':
    main()
