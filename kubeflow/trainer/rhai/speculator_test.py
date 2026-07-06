# Copyright 2025 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for SpeculatorTrainer and CRD conversion."""

import pytest

from kubeflow.trainer.constants import constants
from kubeflow.trainer.rhai.speculator import (
    SpeculatorMode,
    SpeculatorTrainer,
    SpeculatorType,
    _render_speculator_training_script,
    get_trainer_cr_from_speculator_trainer,
)
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.types import types


def test_speculator_trainer_initialization():
    """Test SpeculatorTrainer initialization with default values."""
    print("Executing test: SpeculatorTrainer initialization with defaults")

    trainer = SpeculatorTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
    )

    assert trainer.verifier_name_or_path == "Qwen/Qwen3-8B"
    assert trainer.speculator_type == SpeculatorType.EAGLE3
    assert trainer.mode == SpeculatorMode.TRAIN_ONLY
    assert trainer.hidden_states_path == "/data/hidden_states"
    assert trainer.checkpoint_dir == "/tmp/checkpoints"
    assert trainer.draft_vocab_size is None
    assert trainer.epochs == 3
    assert trainer.lr == 1e-4
    assert trainer.max_samples is None
    assert trainer.num_nodes is None
    assert trainer.resources_per_node is None
    assert trainer.packages_to_install is None
    assert trainer.pip_index_urls == list(constants.DEFAULT_PIP_INDEX_URLS)
    assert trainer.env is None
    assert trainer.output_dir is None
    assert trainer.data_connection_name is None
    assert trainer.enable_progression_tracking is True
    assert trainer.metrics_port == 28080
    assert trainer.metrics_poll_interval_seconds == 30

    print("test execution complete")


def test_speculator_trainer_with_custom_config():
    """Test SpeculatorTrainer with custom configuration."""
    print("Executing test: SpeculatorTrainer with custom configuration")

    trainer = SpeculatorTrainer(
        verifier_name_or_path="meta-llama/Llama-3.1-70B",
        speculator_type=SpeculatorType.EAGLE3,
        mode=SpeculatorMode.TRAIN_ONLY,
        hidden_states_path="/mnt/pvc/hidden_states",
        checkpoint_dir="/mnt/pvc/checkpoints/eagle3",
        draft_vocab_size=32000,
        epochs=5,
        lr=5e-5,
        max_samples=10000,
        num_nodes=2,
        resources_per_node={"nvidia.com/gpu": 4},
        packages_to_install=["speculators"],
        pip_index_urls=["https://custom.pypi.org/simple"],
        env={"WANDB_DISABLED": "true"},
        enable_progression_tracking=True,
        metrics_port=28090,
        metrics_poll_interval_seconds=60,
    )

    assert trainer.verifier_name_or_path == "meta-llama/Llama-3.1-70B"
    assert trainer.speculator_type == SpeculatorType.EAGLE3
    assert trainer.hidden_states_path == "/mnt/pvc/hidden_states"
    assert trainer.checkpoint_dir == "/mnt/pvc/checkpoints/eagle3"
    assert trainer.draft_vocab_size == 32000
    assert trainer.epochs == 5
    assert trainer.lr == 5e-5
    assert trainer.max_samples == 10000
    assert trainer.num_nodes == 2
    assert trainer.resources_per_node == {"nvidia.com/gpu": 4}
    assert trainer.packages_to_install == ["speculators"]
    assert trainer.pip_index_urls == ["https://custom.pypi.org/simple"]
    assert trainer.env == {"WANDB_DISABLED": "true"}
    assert trainer.metrics_port == 28090
    assert trainer.metrics_poll_interval_seconds == 60

    print("test execution complete")


def test_speculator_mode_train_only_requires_hidden_states():
    """Test that TRAIN_ONLY mode requires hidden_states_path."""
    print("Executing test: TRAIN_ONLY mode requires hidden_states_path")

    with pytest.raises(ValueError, match="hidden_states_path is required for TRAIN_ONLY mode"):
        SpeculatorTrainer(
            verifier_name_or_path="Qwen/Qwen3-8B",
            mode=SpeculatorMode.TRAIN_ONLY,
        )

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="DATA_ONLY mode not yet supported",
            expected_status=FAILED,
            config={"mode": SpeculatorMode.DATA_ONLY},
            expected_error=ValueError,
        ),
        TestCase(
            name="OFFLINE mode not yet supported",
            expected_status=FAILED,
            config={"mode": SpeculatorMode.OFFLINE},
            expected_error=ValueError,
        ),
    ],
)
def test_unsupported_mode_validation(test_case):
    """Test that unsupported modes raise ValueError."""
    print(f"Executing test: {test_case.name}")

    try:
        SpeculatorTrainer(
            verifier_name_or_path="Qwen/Qwen3-8B",
            hidden_states_path="/data/hidden_states",
            **test_case.config,
        )

        assert test_case.expected_status == SUCCESS

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="port too low (1023)",
            expected_status=FAILED,
            config={"metrics_port": 1023},
            expected_error=ValueError,
        ),
        TestCase(
            name="port too high (65536)",
            expected_status=FAILED,
            config={"metrics_port": 65536},
            expected_error=ValueError,
        ),
        TestCase(
            name="port minimum boundary (1024)",
            expected_status=SUCCESS,
            config={"metrics_port": 1024},
            expected_output=1024,
        ),
        TestCase(
            name="port maximum boundary (65535)",
            expected_status=SUCCESS,
            config={"metrics_port": 65535},
            expected_output=65535,
        ),
    ],
)
def test_metrics_port_validation(test_case):
    """Test metrics_port validation."""
    print(f"Executing test: {test_case.name}")

    try:
        trainer = SpeculatorTrainer(
            verifier_name_or_path="Qwen/Qwen3-8B",
            hidden_states_path="/data/hidden_states",
            metrics_port=test_case.config["metrics_port"],
        )

        assert test_case.expected_status == SUCCESS
        assert trainer.metrics_port == test_case.expected_output

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="poll interval too low (4 seconds)",
            expected_status=FAILED,
            config={"metrics_poll_interval_seconds": 4},
            expected_error=ValueError,
        ),
        TestCase(
            name="poll interval too high (301 seconds)",
            expected_status=FAILED,
            config={"metrics_poll_interval_seconds": 301},
            expected_error=ValueError,
        ),
        TestCase(
            name="poll interval minimum boundary (5 seconds)",
            expected_status=SUCCESS,
            config={"metrics_poll_interval_seconds": 5},
            expected_output=5,
        ),
        TestCase(
            name="poll interval maximum boundary (300 seconds)",
            expected_status=SUCCESS,
            config={"metrics_poll_interval_seconds": 300},
            expected_output=300,
        ),
    ],
)
def test_metrics_poll_interval_validation(test_case):
    """Test metrics_poll_interval_seconds validation."""
    print(f"Executing test: {test_case.name}")

    try:
        trainer = SpeculatorTrainer(
            verifier_name_or_path="Qwen/Qwen3-8B",
            hidden_states_path="/data/hidden_states",
            metrics_poll_interval_seconds=test_case.config["metrics_poll_interval_seconds"],
        )

        assert test_case.expected_status == SUCCESS
        assert trainer.metrics_poll_interval_seconds == test_case.expected_output

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error

    print("test execution complete")


def test_s3_output_dir_requires_data_connection():
    """Test that S3 output_dir requires data_connection_name."""
    print("Executing test: S3 output_dir requires data_connection_name")

    with pytest.raises(ValueError, match="data_connection_name is required"):
        SpeculatorTrainer(
            verifier_name_or_path="Qwen/Qwen3-8B",
            hidden_states_path="/data/hidden_states",
            output_dir="s3://my-bucket/checkpoints",
        )

    print("test execution complete")


def test_pvc_output_dir_normalized():
    """Test that PVC output_dir is normalized."""
    print("Executing test: PVC output_dir normalization")

    trainer = SpeculatorTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
        output_dir="pvc://my-pvc/checkpoints/",
    )

    assert trainer.output_dir == "pvc://my-pvc/checkpoints"

    print("test execution complete")


def test_training_script_content():
    """Test that generated training script contains expected CLI arguments."""
    print("Executing test: Training script content")

    trainer = SpeculatorTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
        checkpoint_dir="/mnt/checkpoints",
        draft_vocab_size=32000,
        epochs=5,
        lr=1e-5,
    )

    script = _render_speculator_training_script(trainer)

    assert "import sys" in script
    assert "import runpy" in script
    assert 'runpy.run_module("speculators.scripts.train"' in script
    assert "Qwen/Qwen3-8B" in script
    assert "/data/hidden_states" in script
    assert "/mnt/checkpoints" in script
    assert "32000" in script
    assert "eagle3" in script

    print("test execution complete")


def test_training_script_omits_none_optional_fields():
    """Test that generated training script omits optional CLI args when None."""
    print("Executing test: Training script omits None optional fields")

    trainer = SpeculatorTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
    )

    script = _render_speculator_training_script(trainer)

    assert "--draft-vocab-size" not in script

    print("test execution complete")


def test_crd_conversion_train_only():
    """Test CRD conversion for TRAIN_ONLY mode."""
    print("Executing test: CRD conversion for TRAIN_ONLY")

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="registry.redhat.io/rhaii/model-opt-cuda-rhel9:3.5",
        ),
    )

    trainer = SpeculatorTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
        num_nodes=2,
    )

    trainer_crd = get_trainer_cr_from_speculator_trainer(runtime, trainer)

    assert trainer_crd.num_nodes == 2
    assert trainer_crd.command is not None

    script = " ".join(trainer_crd.command) if trainer_crd.command else ""
    assert "speculator_train.py" in script
    assert "Qwen/Qwen3-8B" in script

    print("test execution complete")


def test_crd_conversion_with_env_vars():
    """Test that user env vars appear in CRD."""
    print("Executing test: CRD conversion with env vars")

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="registry.redhat.io/rhaii/model-opt-cuda-rhel9:3.5",
        ),
    )

    trainer = SpeculatorTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
        env={"WANDB_DISABLED": "true", "NCCL_DEBUG": "INFO"},
    )

    trainer_crd = get_trainer_cr_from_speculator_trainer(runtime, trainer)

    assert trainer_crd.env is not None
    env_names = {e.name for e in trainer_crd.env}
    assert "WANDB_DISABLED" in env_names
    assert "NCCL_DEBUG" in env_names

    wandb_env = next(e for e in trainer_crd.env if e.name == "WANDB_DISABLED")
    assert wandb_env.value == "true"

    print("test execution complete")


def test_crd_conversion_no_env_vars():
    """Test CRD conversion when no env vars are provided."""
    print("Executing test: CRD conversion without env vars")

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="registry.redhat.io/rhaii/model-opt-cuda-rhel9:3.5",
        ),
    )

    trainer = SpeculatorTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
    )

    trainer_crd = get_trainer_cr_from_speculator_trainer(runtime, trainer)

    assert trainer_crd.env is None

    print("test execution complete")


def test_crd_conversion_with_resources():
    """Test CRD conversion with resources_per_node."""
    print("Executing test: CRD conversion with resources_per_node")

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="registry.redhat.io/rhaii/model-opt-cuda-rhel9:3.5",
        ),
    )

    trainer = SpeculatorTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
        resources_per_node={"nvidia.com/gpu": 2, "cpu": 8, "memory": "32Gi"},
    )

    trainer_crd = get_trainer_cr_from_speculator_trainer(runtime, trainer)

    assert trainer_crd.resources_per_node is not None

    print("test execution complete")


def test_crd_uses_torchrun_entrypoint():
    """Test that CRD conversion sets torchrun as the entrypoint."""
    print("Executing test: CRD uses torchrun entrypoint")

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="registry.redhat.io/rhaii/model-opt-cuda-rhel9:3.5",
        ),
    )

    trainer = SpeculatorTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
    )

    get_trainer_cr_from_speculator_trainer(runtime, trainer)

    assert runtime.trainer.command == constants.TORCH_COMMAND

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="trainer with progression tracking disabled",
            expected_status=SUCCESS,
            config={"enable_progression_tracking": False},
        ),
        TestCase(
            name="trainer with progression tracking enabled (defaults)",
            expected_status=SUCCESS,
            config={"enable_progression_tracking": True},
        ),
        TestCase(
            name="trainer with custom port",
            expected_status=SUCCESS,
            config={
                "enable_progression_tracking": True,
                "metrics_port": 8888,
            },
        ),
        TestCase(
            name="trainer with long poll interval",
            expected_status=SUCCESS,
            config={
                "enable_progression_tracking": True,
                "metrics_poll_interval_seconds": 120,
            },
        ),
    ],
)
def test_speculator_trainer_configurations(test_case):
    """Test various SpeculatorTrainer configurations."""
    print(f"Executing test: {test_case.name}")

    try:
        trainer = SpeculatorTrainer(
            verifier_name_or_path="Qwen/Qwen3-8B",
            hidden_states_path="/data/hidden_states",
            **test_case.config,
        )

        assert test_case.expected_status == SUCCESS

        for key, value in test_case.config.items():
            assert getattr(trainer, key) == value

    except Exception as e:
        if test_case.expected_error:
            assert type(e) is test_case.expected_error
        else:
            raise

    print("test execution complete")


def test_speculator_enum_values():
    """Test enum values for SpeculatorMode and SpeculatorType."""
    print("Executing test: Enum values")

    assert SpeculatorMode.TRAIN_ONLY.value == "train_only"
    assert SpeculatorMode.DATA_ONLY.value == "data_only"
    assert SpeculatorMode.OFFLINE.value == "offline"
    assert SpeculatorType.EAGLE3.value == "eagle3"
    assert SpeculatorType.DFLASH.value == "dflash"
    assert SpeculatorType.MTP.value == "mtp"
    assert SpeculatorType.PEAGLE.value == "peagle"

    print("test execution complete")
