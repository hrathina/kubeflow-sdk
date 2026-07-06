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

"""SpeculatorTrainer for custom draft model training via the speculators library.

This module provides the SpeculatorTrainer dataclass for training speculative
decoding draft models (e.g., Eagle3) using the speculators library. Currently
supports TRAIN_ONLY mode, which trains from pre-extracted hidden states on a PVC.
"""

from dataclasses import dataclass, field
from enum import Enum

from kubeflow_trainer_api import models

import kubeflow.trainer.backends.kubernetes.utils as k8s_utils
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types


class SpeculatorMode(Enum):
    """Training mode for speculator training.

    Args:
        TRAIN_ONLY: Train draft model from pre-extracted hidden states on PVC.
        DATA_ONLY: Extract hidden states from verifier model via vLLM (future).
        OFFLINE: Train using a user-managed vLLM endpoint (future).
    """

    TRAIN_ONLY = "train_only"
    DATA_ONLY = "data_only"
    OFFLINE = "offline"


class SpeculatorType(Enum):
    """Draft model architecture for speculative decoding.

    Args:
        EAGLE3: Eagle3 draft model architecture.
        DFLASH: DFlash draft model architecture.
        MTP: Multi-Token Prediction draft model architecture.
        PEAGLE: PEAGLE draft model architecture.
    """

    EAGLE3 = "eagle3"
    DFLASH = "dflash"
    MTP = "mtp"
    PEAGLE = "peagle"


_SUPPORTED_MODES = {SpeculatorMode.TRAIN_ONLY}
_SUPPORTED_TYPES = {SpeculatorType.EAGLE3}


@dataclass
class SpeculatorTrainer:
    """RHAI trainer for custom draft model training via the speculators library.

    Args:
        verifier_name_or_path: HuggingFace model ID or path to the verifier model.
        speculator_type: Draft model architecture (default: EAGLE3).
        mode: Training mode (TRAIN_ONLY for this ticket).
        hidden_states_path: Pre-extracted hidden states on PVC (required for TRAIN_ONLY).
        checkpoint_dir: Output directory for the trained draft model.
        draft_vocab_size: Draft model vocabulary size.
        epochs: Training epochs (default: 3).
        lr: Learning rate (default: 1e-4).
        max_samples: Maximum samples from dataset.
        num_nodes: Number of nodes for distributed training.
        resources_per_node: Computing resources per node.
            Example: {"nvidia.com/gpu": 2, "memory": "64Gi", "cpu": "8"}
        packages_to_install: Python packages to install before training.
        pip_index_urls: PyPI index URLs for package installation.
        env: Environment variables to set in training pods.
        output_dir: PVC or S3 URI for checkpoint storage.
        data_connection_name: Kubernetes secret name for S3 credentials.
        enable_progression_tracking: Enable progression tracking.
        metrics_port: HTTP server port for metrics endpoint.
        metrics_poll_interval_seconds: How often controller polls metrics endpoint.
    """

    verifier_name_or_path: str
    speculator_type: SpeculatorType = SpeculatorType.EAGLE3
    mode: SpeculatorMode = SpeculatorMode.TRAIN_ONLY
    hidden_states_path: str | None = None
    checkpoint_dir: str = "/tmp/checkpoints"
    draft_vocab_size: int | None = None
    epochs: int = 3
    lr: float = 1e-4
    max_samples: int | None = None

    num_nodes: int | None = None
    resources_per_node: dict | None = None
    packages_to_install: list[str] | None = None
    pip_index_urls: list[str] = field(
        default_factory=lambda: list(constants.DEFAULT_PIP_INDEX_URLS)
    )
    env: dict[str, str] | None = None
    output_dir: str | None = None
    data_connection_name: str | None = None

    enable_progression_tracking: bool = True
    metrics_port: int = 28080
    metrics_poll_interval_seconds: int = 30

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.mode not in _SUPPORTED_MODES:
            supported = ", ".join(m.value for m in _SUPPORTED_MODES)
            raise ValueError(
                f"Mode '{self.mode.value}' is not yet supported. "
                f"Currently supported modes: {supported}."
            )

        if self.speculator_type not in _SUPPORTED_TYPES:
            supported = ", ".join(t.value for t in _SUPPORTED_TYPES)
            raise ValueError(
                f"Speculator type '{self.speculator_type.value}' is not yet supported. "
                f"Currently supported types: {supported}."
            )

        if self.mode == SpeculatorMode.TRAIN_ONLY and not self.hidden_states_path:
            raise ValueError(
                "hidden_states_path is required for TRAIN_ONLY mode. "
                "Provide the path to pre-extracted hidden states on PVC."
            )

        if not isinstance(self.metrics_port, int):
            raise ValueError(
                f"metrics_port must be an integer, got {type(self.metrics_port).__name__}"
            )
        if self.metrics_port < 1024 or self.metrics_port > 65535:
            raise ValueError(f"metrics_port must be in range 1024-65535, got {self.metrics_port}")

        if not isinstance(self.metrics_poll_interval_seconds, int):
            raise ValueError(
                f"metrics_poll_interval_seconds must be an integer, "
                f"got {type(self.metrics_poll_interval_seconds).__name__}"
            )
        if self.metrics_poll_interval_seconds < 5 or self.metrics_poll_interval_seconds > 300:
            raise ValueError(
                f"metrics_poll_interval_seconds must be in range 5-300 seconds, "
                f"got {self.metrics_poll_interval_seconds}"
            )

        if self.output_dir:
            from kubeflow.trainer.rhai.utils import normalize_and_validate_output_dir

            self.output_dir = normalize_and_validate_output_dir(self.output_dir)

        if (
            self.output_dir
            and self.output_dir.startswith("s3://")
            and not self.data_connection_name
        ):
            raise ValueError(
                "data_connection_name is required when using S3 output_dir. "
                "Provide the name of the Kubernetes secret containing S3 credentials."
            )


def _render_speculator_training_script(trainer: SpeculatorTrainer) -> str:
    """Generate a minimal script that invokes the speculators CLI entry point.

    The SDK user only provides a few fields (verifier, speculator_type, etc.).
    This function builds the corresponding CLI args and delegates to the
    speculators library, which handles all its own defaults internally.

    Args:
        trainer: SpeculatorTrainer configuration.

    Returns:
        Python source code string for the training script.
    """
    cli_args = [
        "train",
        "--verifier-name-or-path",
        trainer.verifier_name_or_path,
        "--speculator-type",
        trainer.speculator_type.value,
        "--save-path",
        trainer.checkpoint_dir,
        "--epochs",
        str(trainer.epochs),
        "--lr",
        str(trainer.lr),
    ]

    if trainer.hidden_states_path is not None:
        cli_args.extend(["--hidden-states-path", trainer.hidden_states_path])

    if trainer.draft_vocab_size is not None:
        cli_args.extend(["--draft-vocab-size", str(trainer.draft_vocab_size)])

    args_repr = repr(cli_args)

    return (
        "import sys\n"
        "import runpy\n"
        "\n"
        f"sys.argv = {args_repr}\n"
        "\n"
        'runpy.run_module("speculators.scripts.train", run_name="__main__")\n'
    )


def _build_install_snippet(
    packages_to_install: list[str] | None,
    pip_index_urls: list[str],
) -> str:
    """Build the shell snippet to install Python packages if requested."""
    if not packages_to_install:
        return ""
    return k8s_utils.get_script_for_python_packages(
        packages_to_install,
        pip_index_urls,
    )


def _get_command_from_runtime(
    runtime: types.Runtime,
    func_code: str,
    func_file: str,
    install_snippet: str,
) -> list[str]:
    """Build command using runtime's command template.

    Args:
        runtime: Runtime configuration with command template.
        func_code: The training function code to execute.
        func_file: The filename to write the code to.
        install_snippet: Package installation script to prepend.

    Returns:
        Command list ready for trainer_crd.command.
    """
    command = []
    for c in runtime.trainer.command:
        if "{func_file}" in c:
            exec_script = c.format(func_code=func_code, func_file=func_file)
            if install_snippet:
                exec_script = install_snippet + exec_script
            command.append(exec_script)
        else:
            command.append(c)
    return command


def get_trainer_cr_from_speculator_trainer(
    runtime: types.Runtime,
    trainer: SpeculatorTrainer,
    initializer: types.Initializer | None = None,
) -> models.TrainerV1alpha1Trainer:
    """Build Trainer CRD for SpeculatorTrainer.

    Args:
        runtime: Runtime configuration.
        trainer: SpeculatorTrainer configuration.
        initializer: Optional initializer configuration.

    Returns:
        Trainer CRD spec.
    """
    runtime.trainer.set_command(constants.TORCH_COMMAND)

    trainer_crd = models.TrainerV1alpha1Trainer()

    if trainer.num_nodes is not None:
        trainer_crd.num_nodes = trainer.num_nodes

    if trainer.resources_per_node:
        trainer_crd.resources_per_node = k8s_utils.get_resources_per_node(
            trainer.resources_per_node
        )

    install_snippet = _build_install_snippet(trainer.packages_to_install, trainer.pip_index_urls)

    func_code = _render_speculator_training_script(trainer)
    func_file = "speculator_train.py"

    trainer_crd.command = _get_command_from_runtime(
        runtime=runtime,
        func_code=func_code,
        func_file=func_file,
        install_snippet=install_snippet,
    )

    trainer_crd.env = (
        [models.IoK8sApiCoreV1EnvVar(name=k, value=v) for k, v in trainer.env.items()]
        if trainer.env
        else None
    )

    return trainer_crd
