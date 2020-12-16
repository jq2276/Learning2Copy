#!/usr/bin/env python

import os
import numpy as np
import torch

from collections import defaultdict
from evaluation.metrics import bleu, distinct


class MetricsManager(object):

    def __init__(self):
        self.metrics_val = defaultdict(float)
        self.metrics_cum = defaultdict(float)
        self.num_samples = 0

    def update(self, metrics):

        num_samples = metrics.pop("num_samples", 1)
        self.num_samples += num_samples

        for key, val in metrics.items():
            if val is not None:
                if isinstance(val, torch.Tensor):
                    val = val.item()
                    self.metrics_cum[key] += val * num_samples
                else:
                    assert len(val) == 2
                    val, num_words = val[0].item(), val[1]
                    self.metrics_cum[key] += np.array(
                        [val * num_samples, num_words])
                self.metrics_val[key] = val

    def get(self, name):
        val = self.metrics_cum.get(name)
        if not isinstance(val, float):
            val = val[0]
        return val / self.num_samples

    def report_val(self):
        metric_strs = []
        for key, val in self.metrics_val.items():
            metric_str = "{}-{:.3f}".format(key.upper(), val)
            metric_strs.append(metric_str)
        metric_strs = "   ".join(metric_strs)
        return metric_strs

    def report_cum(self):
        metric_strs = []
        for key, val in self.metrics_cum.items():
            if isinstance(val, float):
                val, num_words = val, None
            else:
                val, num_words = val

            metric_str = "{}-{:.3f}".format(key.upper(), val / self.num_samples)
            metric_strs.append(metric_str)

            # if num_words is not None:
            #     ppl = np.exp(min(val / num_words, 100))
            #     metric_str = "{}_PPL-{:.3f}".format(key.upper(), ppl)
            #     metric_strs.append(metric_str)

        metric_strs = "   ".join(metric_strs)
        return metric_strs


def evaluate(model, data_iter):
    model.eval()
    mm = MetricsManager()
    with torch.no_grad():
        for inputs in data_iter:
            metrics = model.iterate(inputs=inputs, is_training=False)
            mm.update(metrics)
    return mm


class Trainer(object):
    """
    Trainer
    """
    def __init__(self,
                 model,
                 optimizer,
                 train_iter,
                 valid_iter,
                 logger,
                 generator=None,
                 num_epochs=1,
                 save_dir=None,
                 log_steps=None,
                 valid_steps=None,
                 grad_clip=None,
                 lr_scheduler=None,
                 stage=None):
        self.model = model
        self.optimizer = optimizer
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.logger = logger
        self.generator = generator
        self.is_decreased_valid_metric = True
        self.valid_metric_name = "loss"
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.log_steps = log_steps
        self.valid_steps = valid_steps
        self.grad_clip = grad_clip
        self.lr_scheduler = lr_scheduler
        self.stage = stage
        self.cur_stage = -float("inf")

        self.best_valid_metric = float("inf") if self.is_decreased_valid_metric else -float("inf")
        self.epoch = 0
        self.batch_num = 0

        self.train_start_message = "\n".join(["*************",
                                              " Training: ",
                                              "*************"])
        self.valid_start_message = "\n".join(["*************",
                                              " Evaluating: ",
                                              "*************"])

    def summarize_train_metrics(self, metrics, global_step):
        """
        summarize_train_metrics
        """
        for key, val in metrics.items():
            if isinstance(val, (list, tuple)):
                val = val[0]
            if isinstance(val, torch.Tensor):
                self.train_writer.add_scalar(key, val, global_step)

    def summarize_valid_metrics(self, metrics_mm, global_step):
        """
        summarize_valid_metrics
        """
        for key in metrics_mm.metrics_cum.keys():
            val = metrics_mm.get(key)
            self.valid_writer.add_scalar(key, val, global_step)

    def train_epoch(self):

        self.epoch += 1
        train_mm = MetricsManager()
        num_batches = len(self.train_iter)
        self.logger.info(self.train_start_message)

        for batch_id, inputs in enumerate(self.train_iter, 1):
            self.model.train()
            metrics = self.model.iterate(inputs,
                                         optimizer=self.optimizer,
                                         grad_clip=self.grad_clip,
                                         is_training=True,
                                         epoch=self.epoch)
            train_mm.update(metrics)
            self.batch_num += 1

            if batch_id % self.log_steps == 0:
                message_prefix = "CurEpoch: {}    Batch: {}/{}".format(self.epoch, batch_id, num_batches)
                metrics_message = train_mm.report_val()
                self.logger.info("   ".join(
                    [message_prefix, metrics_message]))

            if batch_id % self.valid_steps == 0:
                self.logger.info(self.valid_start_message)
                valid_mm = evaluate(self.model, self.valid_iter)
                message_prefix = "CurEpoch: {}    Batch: {}/{}".format(self.epoch, batch_id, num_batches)
                metrics_message = valid_mm.report_cum()
                self.logger.info("   ".join([message_prefix, metrics_message]))
                cur_valid_metric = valid_mm.get(self.valid_metric_name)
                if self.is_decreased_valid_metric:
                    is_best = cur_valid_metric < self.best_valid_metric
                else:
                    is_best = cur_valid_metric > self.best_valid_metric

                if self.cur_stage == 0 and self.stage == 1:
                    is_best = True
                    self.cur_stage = 1   # will not be implemented in the next epoch

                if is_best:
                    self.best_valid_metric = cur_valid_metric
                self.save(is_best)
                self.logger.info("*" * 13 + "\n")

        self.save()
        self.logger.info('')

    def train(self):
        valid_mm = evaluate(self.model, self.valid_iter)
        self.logger.info(valid_mm.report_cum())
        for _ in range(self.epoch, self.num_epochs):
            self.train_epoch()

    def save(self, is_best=False):
        """
        save
        """
        if is_best:
            best_model_file = os.path.join(self.save_dir, "best_{}.model".format(self.stage))
            best_train_file = os.path.join(self.save_dir, "best_{}.train".format(self.stage))
            torch.save(self.model.state_dict(), best_model_file)

            train_state = {"epoch": self.epoch,
                           "batch_num": self.batch_num,
                           "best_valid_metric": self.best_valid_metric,
                           "optimizer": self.optimizer.state_dict()}
            if self.lr_scheduler is not None:
                train_state["lr_scheduler"] = self.lr_scheduler.state_dict()
            torch.save(train_state, best_train_file)

            self.logger.info(
                "Saved best model state to '{}' with new best valid metric {}-{:.3f}".format(
                    best_model_file, self.valid_metric_name.upper(), self.best_valid_metric))

    def load(self, file_prefix):
        """
        load
        """

        model_file = "{}.model".format(file_prefix)
        train_file = "{}.train".format(file_prefix)

        self.cur_stage = int(os.path.split(file_prefix)[1].split("_")[-1])
        if self.cur_stage not in [0, 1]:
            raise NotImplementedError

        model_state_dict = torch.load(
            model_file, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(model_state_dict)
        self.logger.info("Loaded model state from '{}'".format(model_file))

        train_state_dict = torch.load(
            train_file, map_location=lambda storage, loc: storage)
        self.epoch = train_state_dict["epoch"]
        self.best_valid_metric = train_state_dict["best_valid_metric"]
        self.batch_num = train_state_dict["batch_num"]
        self.optimizer.load_state_dict(train_state_dict["optimizer"])
        if self.lr_scheduler is not None and "lr_scheduler" in train_state_dict:
            self.lr_scheduler.load_state_dict(train_state_dict["lr_scheduler"])
        self.logger.info(
            "Loaded train state from '{}' with (epoch-{} best_valid_metric-{:.3f})".format(
                train_file, self.epoch, self.best_valid_metric))


def evaluate_generation(generator,
                        data_iter,
                        save_file=None):

    results, preds_all = generator.generate(batch_iter=data_iter)

    refs = [result.tgt.split(" ") for result in results]
    hyps = [result.preds.split(" ") for result in results]

    report_message = []

    avg_len = np.average([len(s) for s in hyps])
    report_message.append("Avg_Len-{:.3f}".format(avg_len))

    bleu_1, bleu_2 = bleu(hyps, refs)
    report_message.append("Bleu-{:.4f}/{:.4f}".format(bleu_1, bleu_2))

    intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(hyps)
    report_message.append("Inter_Dist-{:.4f}/{:.4f}".format(inter_dist1, inter_dist2))

    print("\n".join(report_message))

    if save_file is not None:
        write_results(preds_all, save_file)
        print("Saved generation results to '{}'".format(save_file))


def write_results(results, results_file):
    with open(results_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write("{}\n".format("".join(result)))
