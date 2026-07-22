# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Version-pinned SGLang boundary for offline EAGLE3 data preparation."""

from __future__ import annotations

import logging
from array import array
from types import MethodType
from typing import List, Optional

import torch
import torch.distributed as dist
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.moe.utils import initialize_moe_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_components.dp_attn import prepare_mlp_sync_batch_raw
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import require_mlp_sync, require_mlp_tp_gather

from specforge.distributed import get_tp_group

from .model_runner import SGLangRunner
from .utils import wrap_offline_eagle3_logits_processors

logger = logging.getLogger(__name__)


class OfflineSGLangCaptureBackend:
    """Frozen local target used only to materialize offline EAGLE3 features."""

    def __init__(self, model_runner: SGLangRunner) -> None:
        self.model_runner = model_runner

    @classmethod
    def build(
        cls,
        pretrained_model_name_or_path: str,
        *,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "OfflineSGLangCaptureBackend":
        tp_size = dist.get_world_size(get_tp_group())
        server_args = ServerArgs(
            model_path=pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype if torch_dtype is not None else "auto",
            enable_return_hidden_states=True,
            disable_cuda_graph=True,
            chunked_prefill_size=kwargs.get("max_total_tokens", -1),
            tp_size=tp_size,
            pp_size=1,
            **kwargs,
        )

        tp_rank = dist.get_rank(get_tp_group())
        moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
        model_config = ModelConfig.from_server_args(server_args)
        if getattr(model_config, "is_fp4_experts", False):
            if server_args.moe_runner_backend == "auto":
                server_args.moe_runner_backend = "flashinfer_mxfp4"
            if server_args.moe_runner_backend != "flashinfer_mxfp4":
                raise ValueError(
                    "DeepSeek-V4 FP4 experts require "
                    "moe_runner_backend='flashinfer_mxfp4' for offline capture; "
                    f"got {server_args.moe_runner_backend!r}"
                )
        initialize_moe_config(server_args)
        logger.info(
            "Offline SGLang capture uses moe_runner_backend=%s, "
            "kv_cache_dtype=%s, is_fp4_experts=%s",
            server_args.moe_runner_backend,
            server_args.kv_cache_dtype,
            getattr(model_config, "is_fp4_experts", False),
        )
        model_runner = SGLangRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=torch.cuda.current_device(),
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=server_args.ep_size,
            pp_rank=0,
            pp_size=1,
            server_args=server_args,
            nccl_port=None,
            is_draft_worker=False,
        )
        model_runner.alloc_memory_pool()
        model_runner.init_attention_backends()
        model_runner.init_cuda_graphs()
        wrap_offline_eagle3_logits_processors(model_runner.model)
        return cls(model_runner)

    def set_eagle3_capture_layers(self, layer_ids: Optional[List[int]] = None) -> None:
        model = self.model_runner.model
        if hasattr(model, "set_eagle3_layers_to_capture"):
            model.set_eagle3_layers_to_capture(layer_ids)
            return
        if model.__class__.__name__ == "DeepseekV4ForCausalLM":
            self._set_deepseek_v4_capture_layers(model, layer_ids)
            return
        raise AttributeError(
            f"{model.__class__.__name__} does not support auxiliary hidden-state "
            "capture"
        )

    @staticmethod
    def _set_deepseek_v4_capture_layers(model, layer_ids: Optional[List[int]]) -> None:
        """Capture collapsed mHC outputs from selected DeepSeek-V4 layers."""

        if not layer_ids:
            raise ValueError("DeepSeek-V4 capture requires explicit layer ids")

        decoder = model.model
        selected = [int(layer_id) for layer_id in layer_ids]
        if any(
            layer_id < decoder.start_layer or layer_id >= decoder.end_layer
            for layer_id in selected
        ):
            raise ValueError(
                "DeepSeek-V4 capture layers must belong to the local pipeline "
                f"stage [{decoder.start_layer}, {decoder.end_layer}), got {selected}"
            )

        # Fused mHC defers one layer's hc_post into the next layer, so a normal
        # forward hook would observe an incomplete layer output. Offline capture
        # uses the unfused path to expose the same post-layer streams consumed by
        # mega-dflash before reducing the hc_mult axis.
        decoder.use_fused_mhc_post_pre = False
        for layer in decoder.layers:
            if hasattr(layer, "use_fused_mhc_post_pre"):
                layer.use_fused_mhc_post_pre = False

        captured = {}

        def capture_layer(layer_id):
            def hook(_module, _inputs, output):
                hidden_states = output[0] if isinstance(output, tuple) else output
                if hidden_states.ndim != 3:
                    raise RuntimeError(
                        "DeepSeek-V4 layer capture expected [tokens, hc_mult, hidden], "
                        f"got {tuple(hidden_states.shape)} at layer {layer_id}"
                    )
                captured[layer_id] = hidden_states.mean(dim=1)

            return hook

        for layer_id in selected:
            decoder.layers[layer_id].register_forward_hook(capture_layer(layer_id))

        original_forward = decoder.forward

        def forward_with_aux(_decoder, *args, **kwargs):
            captured.clear()
            output = original_forward(*args, **kwargs)
            missing = [layer_id for layer_id in selected if layer_id not in captured]
            if missing:
                raise RuntimeError(
                    f"DeepSeek-V4 did not execute requested capture layers: {missing}"
                )
            aux_hidden_states = [captured[layer_id] for layer_id in selected]
            return output, aux_hidden_states

        decoder.forward = MethodType(forward_with_aux, decoder)
        model.capture_aux_hidden_states = True

    def _maybe_prepare_mlp_sync_batch(self, batch: ScheduleBatch) -> None:
        if require_mlp_sync(self.model_runner.server_args):
            prepare_mlp_sync_batch_raw(
                batch,
                dp_size=self.model_runner.server_args.dp_size,
                attn_tp_size=1,
                attn_cp_size=getattr(self.model_runner.server_args, "attn_cp_size", 1),
                tp_group=self.model_runner.tp_group,
                get_idle_batch=None,
                disable_cuda_graph=self.model_runner.server_args.disable_cuda_graph,
                require_mlp_tp_gather=require_mlp_tp_gather(
                    self.model_runner.server_args
                ),
                disable_overlap_schedule=self.model_runner.server_args.disable_overlap_schedule,
                offload_tags=set(),
            )

    @torch.no_grad()
    def _forward_extend(self, reqs: list[Req]):
        cache_params = CacheInitParams(
            disable=False,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            page_size=self.model_runner.server_args.page_size,
        )
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=RadixCache(cache_params),
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        self._maybe_prepare_mlp_sync_batch(batch)
        if getattr(batch, "prefill_input_ids_cpu", None) is not None:
            batch.input_ids = batch.prefill_input_ids_cpu.to(
                batch.device, non_blocking=True
            )
            batch.prefill_input_ids_cpu = None
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        forward_batch = ForwardBatch.init_new(batch, self.model_runner)
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        output = self.model_runner.forward(forward_batch)
        return output.logits_output if hasattr(output, "logits_output") else output

    def _clear_pools(self) -> None:
        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()

    @torch.no_grad()
    def capture_eagle3(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ):
        """Capture per-request auxiliary and final hidden states without logits."""

        sampling_params = SamplingParams(temperature=0, max_new_tokens=1, top_k=1)
        reqs: list[Req] = []
        data = []
        input_rows = torch.split(input_ids, 1, dim=0)
        attention_rows = torch.split(attention_mask, 1, dim=0)
        loss_rows = torch.split(loss_mask, 1, dim=0)

        for idx, (input_row, attention_row, loss_row) in enumerate(
            zip(input_rows, attention_rows, loss_rows)
        ):
            req = Req(
                rid=str(idx),
                origin_input_text="",
                origin_input_ids=input_row.view(-1).tolist(),
                sampling_params=sampling_params,
            )
            req.full_untruncated_fill_ids = array("q", req.origin_input_ids)
            req.fill_len = len(req.full_untruncated_fill_ids)
            req.extend_input_len = req.fill_len - len(req.prefix_indices)
            req.logprob_start_len = len(req.origin_input_ids) - 1
            reqs.append(req)
            data.append((input_row, attention_row, loss_row))

        input_lens = [len(req.origin_input_ids) for req in reqs]
        try:
            output = self._forward_extend(reqs)
            aux_hidden_states = getattr(output, "aux_hidden_states", None)
            last_hidden_states = getattr(output, "last_hidden_states", None)
            if aux_hidden_states is None or last_hidden_states is None:
                raise RuntimeError(
                    "SGLang did not return the hidden states required for EAGLE3"
                )
            aux_rows = torch.split(aux_hidden_states, input_lens, dim=0)
            last_rows = torch.split(last_hidden_states, input_lens, dim=0)
        finally:
            self._clear_pools()

        return data, aux_rows, last_rows


__all__ = ["OfflineSGLangCaptureBackend"]
