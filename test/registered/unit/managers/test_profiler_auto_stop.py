"""Bounded captures flush their trace before a client's later stop request."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestProfilerAutoStop(unittest.TestCase):
    def setUp(self):
        self.enterContext(envs.SGLANG_PROFILE_V2.override(False))
        self.enterContext(
            get_parallel().override(
                tp_group=SimpleNamespace(cpu_group=None),
                attn_dp_enabled=False,
                tp_rank=0,
                num_dp_ranks=1,
                pp_size=1,
                moe_ep_size=1,
            )
        )
        self.enterContext(patch("torch.distributed.barrier"))
        self.output_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.forward_ct = 0
        self.manager = SchedulerProfilerManager(get_forward_ct=lambda: self.forward_ct)

    def configure(self, *, start_step=None, by_stage=False):
        result = self.manager._init_profile(
            self.output_dir,
            start_step,
            1,
            ["CPU"],
            False,
            False,
            by_stage,
            "test",
        )
        self.assertTrue(result.success)

    def capture(self, *, by_stage=False):
        self.configure(by_stage=by_stage)
        batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        if by_stage:
            self.manager._profile_batch_predicate(batch)
        else:
            self.assertTrue(self.manager._start_profile().success)
        torch.ones(4).sum()
        self.forward_ct = 2
        self.manager._profile_batch_predicate(batch)
        self.assertFalse(self.manager.profile_in_progress)
        traces = list(Path(self.output_dir).glob("*.trace.json.gz"))
        self.assertEqual(len(traces), 1)
        with gzip.open(traces[0], "rt") as f:
            self.assertTrue(json.load(f)["traceEvents"])

    def test_stop_after_step_bound_keeps_flushed_trace(self):
        self.capture()
        result = self.manager._stop_profile()
        self.assertTrue(result.success, result.message)
        self.assertIn("already stopped", result.message)

    def test_stop_after_stage_bound_succeeds(self):
        self.capture(by_stage=True)
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE, batch_size=lambda: 1)
        self.manager._profile_batch_predicate(batch)
        self.assertTrue(self.manager.profile_in_progress)
        torch.ones(4).sum()
        self.manager._profile_batch_predicate(batch)
        self.assertFalse(self.manager.profile_in_progress)
        self.assertEqual(len(list(Path(self.output_dir).glob("*.trace.json.gz"))), 2)
        self.assertTrue(self.manager._stop_profile().success)

    def test_new_scheduled_capture_does_not_reuse_previous_completion(self):
        self.capture()
        self.configure(start_step=10)
        self.assertFalse(self.manager._stop_profile().success)
        batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        self.forward_ct = 10
        self.manager._profile_batch_predicate(batch)
        self.assertTrue(self.manager.profile_in_progress)
        torch.ones(4).sum()
        self.forward_ct = 11
        self.manager._profile_batch_predicate(batch)
        self.assertFalse(self.manager.profile_in_progress)
        self.assertTrue(self.manager._stop_profile().success)

    def test_stop_without_capture_and_repeated_manual_stop_fail(self):
        self.assertFalse(self.manager._stop_profile().success)
        self.configure()
        self.assertTrue(self.manager._start_profile().success)
        self.assertTrue(self.manager._stop_profile().success)
        self.assertFalse(self.manager._stop_profile().success)


if __name__ == "__main__":
    unittest.main()
