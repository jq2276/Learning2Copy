#!/usr/bin/env python

import os
import json
import logging
import argparse
import torch

from util.corpus import KnowledgeCorpus
from gokc_model import GOKC
from util.engine import Trainer
from generator import TopKGenerator
from util.engine import evaluate, evaluate_generation
from util.misc import str2bool


def model_config():
    """
    model_config
    """
    parser = argparse.ArgumentParser()

    # Data
    data_arg = parser.add_argument_group("Data")
    data_arg.add_argument("--data_dir", type=str, default="./data/")
    data_arg.add_argument("--data_prefix", type=str, default="demo")
    data_arg.add_argument("--save_dir", type=str, default="./models/")
    data_arg.add_argument("--embed_file", type=str, default=None)

    # Network
    net_arg = parser.add_argument_group("Model")
    net_arg.add_argument("--embed_size", type=int, default=300)
    net_arg.add_argument("--hidden_size", type=int, default=800)
    net_arg.add_argument("--bidirectional", type=str2bool, default=True)
    net_arg.add_argument("--max_vocab_size", type=int, default=15000)
    net_arg.add_argument("--min_len", type=int, default=1)
    net_arg.add_argument("--max_len", type=int, default=512)
    net_arg.add_argument("--num_layers", type=int, default=2)
    net_arg.add_argument("--attn", type=str, default='mlp',
                         choices=['none', 'mlp', 'dot', 'general'])
    net_arg.add_argument("--share_vocab", type=str2bool, default=True)
    net_arg.add_argument("--with_bridge", type=str2bool, default=True)
    net_arg.add_argument("--tie_embedding", type=str2bool, default=True)

    # Training / Testing
    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument("--optimizer", type=str, default="Adam")
    train_arg.add_argument("--lr", type=float, default=0.0001)
    train_arg.add_argument("--grad_clip", type=float, default=5.0)
    train_arg.add_argument("--dropout", type=float, default=0.5)
    train_arg.add_argument("--num_epochs", type=int, default=10)
    train_arg.add_argument("--stage", type=int, choices=[0, 1], default=0)
    train_arg.add_argument("--use_embed", type=str2bool, default=True)
    train_arg.add_argument("--use_bow", type=str2bool, default=True)
    train_arg.add_argument("--use_kd", type=str2bool, default=True)
    train_arg.add_argument("--use_posterior", type=str2bool, default=True)  # OFF during the inference stage
    train_arg.add_argument("--force_copy", type=str2bool, default=True)

    # Geneation
    gen_arg = parser.add_argument_group("Generation")
    gen_arg.add_argument("--max_dec_len", type=int, default=30)   # 30
    gen_arg.add_argument("--ignore_unk", type=str2bool, default=True)
    gen_arg.add_argument("--length_average", type=str2bool, default=True)
    gen_arg.add_argument("--gen_file", type=str, default="./models/gen.txt")

    # MISC
    misc_arg = parser.add_argument_group("Misc")
    misc_arg.add_argument("--gpu", type=int, default=0)
    misc_arg.add_argument("--log_steps", type=int, default=70)
    misc_arg.add_argument("--valid_steps", type=int, default=350)
    misc_arg.add_argument("--batch_size", type=int, default=32)   # The actual batch size is batch_size * accum_steps
    misc_arg.add_argument("--ckpt", type=str)
    misc_arg.add_argument("--test", action="store_true")
    misc_arg.add_argument("--interact", action="store_true")

    config = parser.parse_args()

    return config


def main():
    """
    main
    """
    config = model_config()
    config.use_gpu = torch.cuda.is_available() and config.gpu >= 0
    device = config.gpu
    if device >= 0:
        torch.cuda.set_device(device)

    # Data definition
    corpus = KnowledgeCorpus(data_dir=config.data_dir, data_prefix=config.data_prefix,
                             min_freq=0, max_vocab_size=config.max_vocab_size,
                             min_len=config.min_len, max_len=config.max_len,
                             embed_file=config.embed_file, share_vocab=config.share_vocab)

    corpus.load()
    if config.test and config.ckpt:
        corpus.reload(data_type='test')
    train_iter = corpus.create_batches(
        config.batch_size, "train", shuffle=True, device=device)
    valid_iter = corpus.create_batches(
        config.batch_size, "valid", shuffle=False, device=device)
    test_iter = corpus.create_batches(
        config.batch_size, "test", shuffle=False, device=device)

    # GOKC
    model = GOKC(src_vocab_size=corpus.SRC.vocab_size, tgt_vocab_size=corpus.TGT.vocab_size,
                 cue_vocab_size=corpus.CUE.vocab_size, goal_vocab_size=corpus.GOAL.vocab_size,
                 embed_size=config.embed_size, hidden_size=config.hidden_size, padding_idx=corpus.padding_idx,
                 unk_idx=corpus.unk_idx, num_layers=config.num_layers, bidirectional=config.bidirectional,
                 attn_mode=config.attn, with_bridge=config.with_bridge, tie_embedding=config.tie_embedding,
                 dropout=config.dropout, use_gpu=config.use_gpu, use_bow=config.use_bow,
                 use_posterior=config.use_posterior, device=config.gpu,
                 use_kd=config.use_kd, force_copy=config.force_copy, stage=config.stage)

    # Generator definition
    generator = TopKGenerator(model=model, src_field=corpus.SRC, tgt_field=corpus.TGT, cue_field=corpus.CUE,
                              max_length=config.max_dec_len, ignore_unk=config.ignore_unk,
                              length_average=config.length_average, use_gpu=config.use_gpu,
                              dec_embedder=model.dec_embedder, device=config.gpu,
                              tgt_vocab_size=corpus.TGT.vocab_size, force_copy=config.force_copy)

    # Testing
    if config.test and config.ckpt:
        print(model)
        model.load(config.ckpt)
        print("Testing ...")
        metrics = evaluate(model, test_iter)
        print(metrics.report_cum())
        print("Generating ...")
        evaluate_generation(generator, test_iter, save_file=config.gen_file)
    else:
        if config.use_embed and config.embed_file is not None:
            model.encoder.embedder.load_embeddings(
                corpus.SRC.embeddings, scale=0.03)
            model.decoder.embedder.load_embeddings(
                corpus.TGT.embeddings, scale=0.03)
        # Optimizer definition
        optimizer = getattr(torch.optim, config.optimizer)(
            model.parameters(), lr=config.lr)

        # Save directory
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)

        # Logger definition
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
        fh = logging.FileHandler(os.path.join(config.save_dir, "train.log"))
        logger.addHandler(fh)

        # Save config
        params_file = os.path.join(config.save_dir, "params.json")
        with open(params_file, 'w') as fp:
            json.dump(config.__dict__, fp, indent=4, sort_keys=True)
        print("Saved params to '{}'".format(params_file))
        logger.info(model)

        # Train
        logger.info("Training starts ...")
        trainer = Trainer(model=model, optimizer=optimizer, train_iter=train_iter,
                          valid_iter=valid_iter, logger=logger, generator=generator,
                          num_epochs=config.num_epochs,
                          save_dir=config.save_dir, log_steps=config.log_steps,
                          valid_steps=config.valid_steps, grad_clip=config.grad_clip,
                          stage=config.stage)
        if config.ckpt is not None:
            trainer.load(file_prefix=config.ckpt)

        # start training
        trainer.train()
        logger.info("Training done!")
        # Test
        if config.stage == 1:
            logger.info("")
            trainer.load(os.path.join(config.save_dir, "best_{}".format(config.stage)))
            logger.info("Testing starts ...")
            metrics = evaluate(model, test_iter)
            logger.info(metrics.report_cum())
            logger.info("Generation starts ...")
            test_gen_file = os.path.join(config.save_dir, "test.result")
            evaluate_generation(generator, test_iter, save_file=test_gen_file)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
