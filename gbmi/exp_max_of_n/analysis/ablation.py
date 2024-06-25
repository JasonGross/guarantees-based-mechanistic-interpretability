from typing import Union, Optional, Tuple, Sequence
from collections import defaultdict
from dataclasses import dataclass
import time
import torch
import numpy as np
from jaxtyping import Float, Integer
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from gbmi.utils import ein
from gbmi.analysis_tools.utils import data_summary
from gbmi.utils.sequences import generate_all_sequences


@dataclass
class AblationOptions:
    EU: bool = False
    EQKE: bool = False
    EQKP: bool = False
    EVOU: bool = False
    PVOU: bool = False

    def short_description(self, latex: bool = False) -> str:
        ablate_set = [
            s
            for s, b in [
                ("EU", self.EU),
                ("EQKE", self.EQKE),
                ("EQKP", self.EQKP),
                ("EVOU", self.EVOU),
                ("PVOU", self.PVOU),
            ]
            if b
        ]
        if ablate_set:
            if latex:
                return "Ablate" + "".join(ablate_set)
            else:
                return "Ablate" + "-".join(ablate_set)
        else:
            return "AblateNone"

    def __hash__(self) -> int:
        return hash((self.EU, self.EQKE, self.EQKP, self.EVOU, self.PVOU))


@torch.no_grad()
def compute_ablations(
    model: HookedTransformer,
    max_incorrect_sequences: int = 64,
    pbar: Optional[tqdm] = None,
) -> Tuple[dict[AblationOptions, dict[str, Union[float, Sequence[int]]]], float]:
    """
    Computes

    Args:
        model: The model to compute the ablation scores for.

    Returns:
        For each option of ablation, a dictionary containing:
        - "loss" (float): The average loss of the model with the ablation.
        - "accuracy" (float): The average accuracy of the model with the ablation.
        - "num_incorrect_sequences" (int): The number of incorrect sequences.
        - "incorrect_sequences" (List[int]): The incorrect sequences, if there are fewer than max_incorrect_sequences of them
    """

    def normalize_result(result):
        # to allow pickling
        if isinstance(result, dict):
            return {k: normalize_result(v) for k, v in result.items()}
        elif isinstance(result, torch.Tensor) and np.prod(result.shape) == 1:
            return result.item()
        elif isinstance(result, float) or isinstance(result, int):
            return result
        else:
            raise TypeError(f"Unexpected type {type(result)} ({result})")
            return result

    start = time.time()
    E, Q, K, V, O, U, P = (
        model.W_E,
        model.W_Q[0, 0],
        model.W_K[0, 0],
        model.W_V[0, 0],
        model.W_O[0, 0],
        model.W_U,
        model.W_pos,
    )
    assert model.cfg.n_heads == 1, model.cfg.n_heads
    assert model.cfg.n_layers == 1, model.cfg.n_layers
    # check all biases 0
    for name, param in model.named_parameters():
        if "b_" in name:
            assert (param == 0).all(), name
    # assert no LN
    assert model.cfg.normalization_type is None, model.cfg.normalization_type
    EVOU = E @ V @ O @ U
    PVOU = P @ V @ O @ U
    Pbar = P.mean(dim=0, keepdim=True)
    Pq = P[-1].unsqueeze(0)
    Phat = P - Pbar
    Ebar = E + Pbar
    Eq = E + Pq
    EQKE = Eq @ Q @ K.T @ Ebar.T / model.blocks[0].attn.attn_scale
    EQKP = Eq @ Q @ K.T @ Phat.T / model.blocks[0].attn.attn_scale
    EU = Eq @ U
    d_vocab, n_ctx = model.cfg.d_vocab, model.cfg.n_ctx
    all_sequences = generate_all_sequences(d_vocab, n_ctx - 1)
    pre_query_maxes = all_sequences.max(dim=-1).values
    all_sequences = torch.cat(
        [all_sequences, torch.zeros(all_sequences.shape[0], 1).long()], dim=1
    )
    result = defaultdict(lambda: defaultdict(float))
    #
    only_EU_count_correct = sum(
        EU[qtok].argmax() ** (n_ctx - 1)
        for qtok in range(d_vocab)
        if EU[qtok].argmax() >= qtok
    )
    only_EU_acc = only_EU_count_correct / d_vocab**n_ctx
    max_tok_range = range(d_vocab)
    if pbar is None:
        max_tok_range = tqdm(max_tok_range, desc="maxtok for EU ablation")
    only_EU_loss = (
        EU[0].softmax(dim=-1)[0]
        + sum(
            (maxtok**n_ctx - (maxtok - 1) ** n_ctx) * EU[qtok].softmax(dim=-1)[maxtok]
            for maxtok in max_tok_range
            for qtok in range(maxtok + 1)
        )
    ) / d_vocab**n_ctx
    only_EU_result = {
        "loss": only_EU_loss,
        "accuracy": only_EU_acc,
        "num_correct_sequences": only_EU_count_correct,
    }
    ablate_all_result = {
        "loss": torch.zeros(d_vocab).softmax(dim=-1)[0],
        "accuracy": 0.0,
        "num_correct_sequences": 0,
    }
    for ablate_EQKE in [False, True]:
        for ablate_EQKP in [False, True]:
            result[
                AblationOptions(
                    EU=False, EVOU=True, PVOU=True, EQKE=ablate_EQKE, EQKP=ablate_EQKP
                )
            ] = only_EU_result
            result[
                AblationOptions(
                    EU=True, EVOU=True, PVOU=True, EQKE=ablate_EQKE, EQKP=ablate_EQKP
                )
            ] = ablate_all_result
    qtok_range = range(d_vocab)
    if pbar is None:
        qtok_range = tqdm(qtok_range, desc="qtok")
    for qtok in qtok_range:
        if pbar is not None:
            pbar.update(1)
        qEQKE, qEQKP, qEU = EQKE[qtok], EQKP[qtok], EU[qtok]
        all_sequences[:, -1] = qtok
        maxes = pre_query_maxes.clone()
        maxes[maxes < qtok] = qtok
        EQKEs = qEQKE[all_sequences]
        EQKPs = qEQKP.clone()
        EQKEs = EQKEs - EQKEs.max(dim=-1, keepdim=True).values
        EQKPs = EQKPs - EQKPs.max(dim=-1, keepdim=True).values
        EQKEs = EQKEs.exp()
        EQKPs = EQKPs.exp()
        EQKEPs = EQKEs * EQKPs
        EQKEsum = EQKEs.sum(dim=-1, keepdim=True)
        EQKPsum = EQKPs.sum(dim=-1, keepdim=True)
        EQKEPsum = EQKEPs.sum(dim=-1, keepdim=True)
        EQKEs = EQKEs / EQKEsum
        EQKPs = EQKPs / EQKPsum
        EQKEPs = EQKEPs / EQKEPsum
        # EVOU -= EVOU[:, qtok].unsqueeze(-1)
        # PVOU -= PVOU[:, qtok].unsqueeze(-1)
        # EU -= EU[:, qtok].unsqueeze(-1)
        for ablate_EQKE in [False, True]:
            for ablate_EQKP in [False, True]:
                if not ablate_EQKE and not ablate_EQKP:
                    attn = EQKEPs
                elif not ablate_EQKE:
                    attn = EQKEs
                elif not ablate_EQKP:
                    attn = EQKPs.unsqueeze(0)
                else:
                    attn = torch.zeros_like(EQKPs).softmax(dim=-1).unsqueeze(0)
                attnV = (EVOU[all_sequences, :] * attn.unsqueeze(-1)).sum(dim=-2)
                attnP = (PVOU * attn.unsqueeze(-1)).sum(dim=-2)
                # attnV = attnV - attnV[torch.arange(attnV.shape[0]), maxes].unsqueeze(-1)
                # attnP = attnP - attnP[torch.arange(attnP.shape[0]), maxes].unsqueeze(-1)
                # EU = EU - EU[qtok].unsqueeze(-1)
                for ablate_EVOU in [False, True]:
                    for ablate_PVOU in [False, True]:
                        if ablate_EVOU and ablate_PVOU:
                            continue
                        for ablate_EU in [False, True]:
                            val = torch.zeros_like(attnV)
                            if not ablate_EVOU:
                                val += attnV
                            if not ablate_PVOU:
                                val += attnP
                            if not ablate_EU:
                                val += qEU
                            val = val.softmax(dim=-1)
                            losses = val[torch.arange(val.shape[0]), maxes]
                            count_correct = (val.argmax(dim=-1) == maxes).sum()
                            opts = AblationOptions(
                                EU=ablate_EU,
                                EVOU=ablate_EVOU,
                                PVOU=ablate_PVOU,
                                EQKE=ablate_EQKE,
                                EQKP=ablate_EQKP,
                            )
                            result[opts]["loss"] += losses.sum() / d_vocab**n_ctx
                            result[opts]["accuracy"] += count_correct / d_vocab**n_ctx
                            result[opts]["num_correct_sequences"] += count_correct
    for k, v in result.items():
        v["num_incorrect_sequences"] = d_vocab**n_ctx - v["num_correct_sequences"]
    end = time.time() - start
    return normalize_result(result), end


def latexify_ablation_results(
    ablation_results: dict[AblationOptions, dict[str, Union[float, Sequence[int]]]],
    float_postfix: str = "Float",
    int_postfix: str = "",
) -> dict[str, float]:
    latex_values = {}
    summary_lists = defaultdict(list)
    for k in sorted(ablation_results.keys(), key=AblationOptions.short_description):
        d = ablation_results[k]
        for key in sorted(d.keys()):
            if isinstance(d[key], int):
                postfix = int_postfix
            elif isinstance(d[key], float):
                postfix = float_postfix
            else:
                raise TypeError((key, type(d[key]), d[key]))
            value_key = "".join(
                v.capitalize() if v[0] != v[0].capitalize() else v
                for v in key.replace("_", "-").split("-")
            )
            latex_key = f"{k.short_description(latex=True)}{value_key}{postfix}"
            latex_values[latex_key] = d[key]
            if k.EQKE and k.EVOU:
                summary_lists[f"AblateAllImportant{value_key}"].append(d[key])
            if k.EQKE or k.EVOU:
                summary_lists[f"AblateAnyImportant{value_key}"].append(d[key])
            else:
                summary_lists[f"AblateOnlyNoise{value_key}"].append(d[key])
            for kind_descr, ablated in (
                ("EQ", k.EU),
                ("EQKE", k.EQKE),
                ("EQKP", k.EQKP),
                ("EV", k.EVOU),
                ("PV", k.PVOU),
            ):
                if ablated:
                    summary_lists[f"Ablate{kind_descr}Plus{value_key}"].append(d[key])
                else:
                    summary_lists[f"Ablate{kind_descr}Complement{value_key}"].append(
                        d[key]
                    )
    for k, v in summary_lists.items():
        latex_values |= data_summary(
            v, prefix=k, float_postfix=float_postfix, int_postfix=int_postfix
        )
    return latex_values
