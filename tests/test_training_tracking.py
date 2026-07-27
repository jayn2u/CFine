import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


fake_config = types.ModuleType("config")
fake_config.data_config = Mock()
fake_config.network_config = Mock()
fake_config.get_image_unique = Mock()
fake_config.log_config = Mock()
fake_config.dir_config = Mock()

fake_test = types.ModuleType("test")
fake_test.test = Mock()

fake_crloss = types.ModuleType("utils.CRLoss")
fake_crloss.CRLoss = Mock()

fake_solver = types.ModuleType("solver")
fake_solver.WarmupMultiStepLR = Mock()

with patch.dict(
    sys.modules,
    {
        "config": fake_config,
        "test": fake_test,
        "utils.CRLoss": fake_crloss,
        "solver": fake_solver,
    },
):
    import train as train_module


class RecordingSession:
    enabled = True

    def __init__(self):
        self.payloads = []
        self.summary = {}
        self.finished = False

    def log(self, payload, step=None):
        self.payloads.append(payload)

    def set_summary(self, metrics):
        self.summary.update(metrics)

    def finish(self):
        self.finished = True


class FakeLoss:
    W = torch.tensor([1.0])

    def parameters(self):
        return []


class FakeDataParallel:
    def __init__(self, module):
        self.module = module

    def cuda(self):
        return self


class FakeNetwork:
    def train(self):
        return self

    def state_dict(self):
        return {}


class FakeOptimizer:
    param_groups = [{"lr": 1e-4}]

    def state_dict(self):
        return {}


class FakeScheduler:
    def step(self):
        pass


class TrainingTrackingTest(unittest.TestCase):
    def test_one_epoch_logs_train_validation_and_best_summary(self):
        session = RecordingSession()
        args = SimpleNamespace(
            image_dir="/dataset/images",
            anno_dir="/dataset/annotations",
            checkpoint_dir=tempfile.gettempdir(),
            batch_size=32,
            num_epoches=1,
            seed=42,
            resume=False,
            model_path=None,
        )
        meters = {
            "loss": SimpleNamespace(avg=3.0, count=32),
            "cmpm_loss": SimpleNamespace(avg=1.0, count=32),
            "cmpc_loss": SimpleNamespace(avg=1.5, count=32),
            "sim_loss": SimpleNamespace(avg=0.05, count=32),
            "image_acc": SimpleNamespace(avg=0.7, count=32),
            "text_acc": SimpleNamespace(avg=0.8, count=32),
        }
        val_metrics = {
            "t2i_R1": 69.5,
            "t2i_R5": 85.0,
            "t2i_R10": 90.0,
            "t2i_mAP": 60.0,
            "t2i_mINP": 45.0,
            "i2t_R1": 81.0,
            "i2t_R5": 95.0,
            "i2t_R10": 97.0,
            "i2t_mAP": 70.0,
            "i2t_mINP": 55.0,
        }

        def network_config(*unused_args, **unused_kwargs):
            args.start_epoch = 0
            return FakeNetwork(), FakeOptimizer()

        with (
            patch.object(train_module, "set_seed"),
            patch.object(train_module, "data_config", return_value=[]),
            patch.object(train_module, "get_image_unique", return_value=[]),
            patch.object(train_module, "Loss", return_value=FakeLoss()),
            patch.object(train_module, "CRLoss", return_value=object()),
            patch.object(train_module.nn, "DataParallel", FakeDataParallel),
            patch.object(train_module, "network_config", side_effect=network_config),
            patch.object(
                train_module,
                "WarmupMultiStepLR",
                return_value=FakeScheduler(),
            ),
            patch.object(train_module, "train", return_value=meters),
            patch.object(train_module, "test", return_value=val_metrics),
            patch.object(train_module, "save_checkpoint"),
            patch.object(
                train_module,
                "start_measurement",
                side_effect=[10.0, 20.0],
                create=True,
            ),
            patch.object(
                train_module,
                "finish_cuda_timer",
                side_effect=[5.0, 2.0],
                create=True,
            ),
            patch.object(
                train_module,
                "get_peak_vram_metrics",
                side_effect=[
                    {"peak_vram_allocated_mb": 9000.0},
                    {"peak_vram_allocated_mb": 7000.0},
                ],
                create=True,
            ),
        ):
            train_module.main(args, wandb_session=session)

        train_payload, val_payload = session.payloads
        self.assertEqual(train_payload["train/loss"], 3.0)
        self.assertEqual(train_payload["train/examples_per_second"], 6.4)
        self.assertEqual(val_payload["val/t2i_R1"], 69.5)
        self.assertEqual(val_payload["val/t2i_error@1"], 30.5)
        self.assertEqual(session.summary["val/best_t2i_R1"], 69.5)
        self.assertEqual(session.summary["val/best_epoch"], 1)
        self.assertTrue(session.finished)


if __name__ == "__main__":
    unittest.main()
