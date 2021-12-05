# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from fairseq.data.dictionary import Dictionary

import pdb

@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "alpha for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )

    sentence_avg: bool = II("optimization.sentence_avg")
    
    joint_vocab_dir: str = field(
        default="dir",
        metadata={"help":"the joined vocab"}
    )

    tgt_vocab_dir: str = field(
        default="dir",
        metadata={"help":"the target vocab"}
    )

    src_vocab_dir: str = field(
        default="dir",
        metadata={"help":"the source vocab"}
    )

    lp_beta: float = field(
        default=0.0,
        metadata={"help": "beta for label smoothing"},
    )
    lp_gamma: float = field(
        default=0.0,
        metadata={"help": "gamma for label smoothing"},
    )
    lp_eps: float = field(
        default=0.0,
        metadata={"help": "eps for label smoothing"},
    )

def divide_s_c_t(shared:Dictionary,source:Dictionary,target:Dictionary):
    res = {"source":[],"common":[],"target":[]}

    for i,j in enumerate(shared):
        if i>=len(shared):
            break
        if j in source and j in target:
            res['common'].append(i)
        elif j not in source:
            res['target'].append(i)
        elif j not in target:
            res['source'].append(i)

    return res



def filter_invalid_token(shared:Dictionary,tgt:Dictionary):
    ret = []
    for i,j in enumerate(shared):
        if i>=len(shared):
            break
        if j in tgt:
            ret.append(i)

    return ret

def label_smooth_lexical_prior_nll_loss(lprobs, target, alpha, beta, gamma, epsilon, ignore_index=None, venn_set=None, reduce=True):
    total = beta+gamma+epsilon
    eps_t = (alpha*beta/total)/len(venn_set['target'])
    eps_c = (alpha*gamma/total)/len(venn_set['common'])
    eps_s = (alpha*epsilon/total)/len(venn_set['source'])

    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)

    assert venn_set is not None

    target_probs = lprobs.t()[venn_set['target']]
    target_probs = target_probs.t()
    target_smooth_loss = -target_probs.sum(dim=-1, keepdim=True)

    common_probs = lprobs.t()[venn_set['common']]
    common_probs = common_probs.t()
    common_smooth_loss = -common_probs.sum(dim=-1, keepdim=True)

    source_probs = lprobs.t()[venn_set['source']]
    source_probs = source_probs.t()
    source_smooth_loss = -source_probs.sum(dim=-1, keepdim=True)


    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        target_smooth_loss.masked_fill_(pad_mask, 0.0)
        common_smooth_loss.masked_fill_(pad_mask, 0.0)
        source_smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        target_smooth_loss = target_smooth_loss.squeeze(-1)
        common_smooth_loss = common_smooth_loss.squeeze(-1)
        source_smooth_loss = source_smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        target_smooth_loss = target_smooth_loss.sum()
        common_smooth_loss = common_smooth_loss.sum()
        source_smooth_loss = source_smooth_loss.sum()
    
    loss = (1.0 - alpha - eps_t) * nll_loss + eps_t * target_smooth_loss + eps_c* common_smooth_loss+eps_s*source_smooth_loss 

    return loss, nll_loss





def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None,mask_labels=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)

    if mask_labels is not None:
        temp_probs = lprobs.t()[mask_labels]
        temp_probs = temp_probs.t()
        smooth_loss = -temp_probs.sum(dim=-1, keepdim=True)

    else:
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
         

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    
    eps_i = epsilon / (len(mask_labels)-1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss

    return loss, nll_loss


@register_criterion(
    "weighted_label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        lp_beta,
        lp_gamma,
        lp_eps,
        src_vocab_dir,
        joint_vocab_dir,
        tgt_vocab_dir,
        ignore_prefix_size=0,
        report_accuracy=False,

    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.alpha = label_smoothing

        self.beta = lp_beta
        self.gamma = lp_gamma
        self.epsilon = lp_eps

        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

        self.src_dict = Dictionary().load(src_vocab_dir)
        self.tgt_dict = Dictionary().load(tgt_vocab_dir)
        self.joint_dict = Dictionary().load(joint_vocab_dir)

        

        self.masked_labels = filter_invalid_token(self.joint_dict,self.tgt_dict)

        self.venn_set = divide_s_c_t(self.joint_dict,self.src_dict,self.tgt_dict)



        

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample) # lprobs are the probs generated by model for given target(grountruth)

        # loss, nll_loss = label_smoothed_nll_loss(
        #     lprobs,
        #     target,
        #     self.eps,
        #     ignore_index=self.padding_idx,
        #     mask_labels=self.masked_labels,
        #     reduce=reduce,
        # )


        loss, nll_loss = label_smooth_lexical_prior_nll_loss(
            lprobs,
            target,
            self.alpha,
            self.beta,
            self.gamma,self.epsilon,
            ignore_index=self.padding_idx,
            venn_set=self.venn_set,
            reduce=True
        )

        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
