import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from models.clip import Transformer
from models.layers import ResidualAttention
from models.pytorch_transformers.modeling_bert import BertConfig, BertEncoder
from train_config import parse_args
from utils.training import (
    build_ema_model,
    get_autocast_dtype,
    optimizer_step,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeScaler:
    def __init__(self, old_scale, new_scale):
        self.old_scale = old_scale
        self.new_scale = new_scale
        self.updated = False

    def scale(self, loss):
        return loss

    def get_scale(self):
        return self.new_scale if self.updated else self.old_scale

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        self.updated = True


class TrainingFeatureTest(unittest.TestCase):
    def test_training_features_are_opt_in(self):
        with patch.object(sys, "argv", ["train.py"]):
            args = parse_args()

        self.assertFalse(args.amp)
        self.assertEqual(args.amp_dtype, "fp16")
        self.assertFalse(args.ema)
        self.assertEqual(args.ema_decay, 0.999)
        self.assertFalse(args.gradient_checkpointing)

    def test_training_features_can_be_enabled(self):
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--amp",
                "--amp_dtype",
                "bf16",
                "--ema",
                "--ema_decay",
                "0.995",
                "--gradient_checkpointing",
            ],
        ):
            args = parse_args()

        self.assertTrue(args.amp)
        self.assertEqual(args.amp_dtype, "bf16")
        self.assertTrue(args.ema)
        self.assertEqual(args.ema_decay, 0.995)
        self.assertTrue(args.gradient_checkpointing)

    def test_ema_decay_must_be_between_zero_and_one(self):
        for value in ("0", "1", "-0.1", "1.1"):
            with (
                self.subTest(value=value),
                patch.object(sys, "argv", ["train.py", "--ema_decay", value]),
                patch("sys.stderr"),
                self.assertRaises(SystemExit),
            ):
                parse_args()

    def test_amp_dtype_names_map_to_torch_dtypes(self):
        self.assertIs(get_autocast_dtype("fp16"), torch.float16)
        self.assertIs(get_autocast_dtype("bf16"), torch.bfloat16)

    def test_optimizer_step_reports_gradient_scaler_overflow(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)

        succeeded = optimizer_step(
            parameter.square(),
            optimizer,
            FakeScaler(old_scale=8.0, new_scale=4.0),
        )

        self.assertFalse(succeeded)

    def test_ema_averages_parameters_and_buffers(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(1, 1, bias=False),
            torch.nn.BatchNorm1d(1),
        )
        model[0].weight.data.fill_(1.0)
        model[1].running_mean.fill_(1.0)
        ema_model = build_ema_model(model, decay=0.5)

        model[0].weight.data.fill_(3.0)
        model[1].running_mean.fill_(3.0)
        ema_model.update_parameters(model)
        model[0].weight.data.fill_(5.0)
        model[1].running_mean.fill_(5.0)
        ema_model.update_parameters(model)

        self.assertEqual(ema_model.module[0].weight.item(), 4.0)
        self.assertEqual(ema_model.module[1].running_mean.item(), 4.0)

    def test_checkpointed_transformer_matches_regular_forward_and_backward(self):
        torch.manual_seed(0)
        regular = Transformer(width=8, layers=2, heads=1)
        checkpointed = Transformer(width=8, layers=2, heads=1)
        checkpointed.load_state_dict(regular.state_dict())
        checkpointed.set_gradient_checkpointing(True)
        regular.train()
        checkpointed.train()
        regular_input = torch.randn(3, 2, 8, requires_grad=True)
        checkpointed_input = regular_input.detach().clone().requires_grad_(True)

        regular_output = regular(regular_input)[0]
        checkpointed_output = checkpointed(checkpointed_input)[0]
        regular_output.sum().backward()
        checkpointed_output.sum().backward()

        torch.testing.assert_close(checkpointed_output, regular_output)
        torch.testing.assert_close(checkpointed_input.grad, regular_input.grad)

    def test_checkpointed_cross_attention_matches_regular_forward_and_backward(self):
        torch.manual_seed(0)
        regular = ResidualAttention(1, 8, 2, att_type="cross")
        checkpointed = ResidualAttention(1, 8, 2, att_type="cross")
        checkpointed.load_state_dict(regular.state_dict())
        checkpointed.set_gradient_checkpointing(True)
        regular.train()
        checkpointed.train()
        regular_x = torch.randn(2, 3, 8, requires_grad=True)
        checkpointed_x = regular_x.detach().clone().requires_grad_(True)
        regular_y = torch.randn(2, 4, 8, requires_grad=True)
        checkpointed_y = regular_y.detach().clone().requires_grad_(True)

        regular_output = regular(regular_x, regular_y)
        checkpointed_output = checkpointed(checkpointed_x, checkpointed_y)
        regular_output.sum().backward()
        checkpointed_output.sum().backward()

        torch.testing.assert_close(checkpointed_output, regular_output)
        torch.testing.assert_close(checkpointed_x.grad, regular_x.grad)
        torch.testing.assert_close(checkpointed_y.grad, regular_y.grad)

    def test_checkpointed_bert_encoder_matches_regular_forward_and_backward(self):
        torch.manual_seed(0)
        config = BertConfig(
            vocab_size_or_config_json_file=32,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        regular = BertEncoder(config)
        checkpointed = BertEncoder(config)
        checkpointed.load_state_dict(regular.state_dict())
        checkpointed.gradient_checkpointing = True
        regular.train()
        checkpointed.train()
        regular_input = torch.randn(2, 4, 8, requires_grad=True)
        checkpointed_input = regular_input.detach().clone().requires_grad_(True)
        attention_mask = torch.zeros(2, 1, 1, 4)
        head_mask = [None, None]

        regular_output = regular(regular_input, attention_mask, head_mask)[0][0]
        checkpointed_output = checkpointed(
            checkpointed_input,
            attention_mask,
            head_mask,
        )[0][0]
        regular_output.sum().backward()
        checkpointed_output.sum().backward()

        torch.testing.assert_close(checkpointed_output, regular_output)
        torch.testing.assert_close(checkpointed_input.grad, regular_input.grad)


class LauncherTest(unittest.TestCase):
    def test_amp_ema_gc_launcher_passes_the_strategy_flags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            args_path = temp_path / "args"
            fake_uv = temp_path / "uv"
            fake_uv.write_text(
                "#!/bin/sh\n"
                'printf "%s\\n" "$@" > "$CAPTURE_ARGS"\n',
                encoding="utf-8",
            )
            fake_uv.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{temp_path}{os.pathsep}{env['PATH']}"
            env["CAPTURE_ARGS"] = str(args_path)

            subprocess.run(
                [str(REPO_ROOT / "run_cfine_amp_ema_gc.sh")],
                cwd=REPO_ROOT,
                env=env,
                check=True,
            )

            args = args_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(args[:3], ["run", "python", "train.py"])
        self.assertIn("--amp", args)
        self.assertIn("--gradient_checkpointing", args)
        self.assertIn("--ema", args)
        self.assertEqual(args[args.index("--ema_decay") + 1], "0.999")


if __name__ == "__main__":
    unittest.main()
