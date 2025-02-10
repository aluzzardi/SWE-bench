# This file contains logic for running evaluations with Dagger: <https://dagger.io/>.

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from logging import Logger
from pathlib import Path

import anyio
import dagger
from anyio import to_thread
from dagger import dag, ReturnType

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    RUN_EVALUATION_LOG_DIR,
)
from swebench.harness.constants.constants import SWEbenchInstance
from swebench.harness.docker_build import setup_logger
from swebench.harness.grading import get_eval_report
from swebench.harness.reporting import make_run_report
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec
from swebench.harness.utils import EvaluationError

RUN_LOG_FILE = "run_instance.log"
TEST_LOG_FILE = "test_output.txt"
PATCH_FILE = "patch.diff"
REPORT_FILE = "report.json"


@dataclass
class Instance:
    run_id: str
    test_spec: TestSpec
    pred: dict

    @property
    def id(self) -> str:
        return self.test_spec.instance_id

    @property
    def log_dir(self) -> Path:
        return (
            RUN_EVALUATION_LOG_DIR
            / self.run_id
            / self.pred.get("model_name_or_path", "None").replace("/", "__")
            / self.id
        )

    async def setup_logger(self) -> Logger:
        setup = functools.partial(
            setup_logger,
            self.id,
            self.log_dir / RUN_LOG_FILE,
            add_stdout=True,
        )
        _logger = await to_thread.run_sync(setup)
        return _logger


def get_instance_image(test_spec: TestSpec) -> dagger.Container:
    ctr = (
        dag.container()
        .from_("ubuntu:22.04")
        .with_exec(["apt", "update"])
        .with_env_variable("DEBIAN_FRONTEND", "noninteractive")
        .with_env_variable("TZ", "Etc/UTC")
        .with_exec(
            [
                "apt",
                "install",
                "-y",
                "wget",
                "git",
                "build-essential",
                "libffi-dev",
                "libtiff-dev",
                "jq",
                "curl",
                "locales",
                "locales-all",
                "tzdata",
            ]
        )
        # .with_exec(["wget", "https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-2-Linux-x86_64.sh", "-O", "miniconda.sh"])
        .with_exec(
            [
                "wget",
                "https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-2-Linux-aarch64.sh",
                "-O",
                "miniconda.sh",
            ]
        )
        .with_exec(["bash", "miniconda.sh", "-b", "-p", "/opt/miniconda3"])
        .with_exec(
            ["bash", "-c", "echo 'export PATH=/opt/miniconda3/bin:$PATH' >> ~/.bashrc"]
        )
        .with_exec(["/opt/miniconda3/bin/conda", "init", "--all"])
        .with_exec(
            [
                "/opt/miniconda3/bin/conda",
                "config",
                "--append",
                "channels",
                "conda-forge",
            ]
        )
        .with_exec(["adduser", "--disabled-password", "--gecos", "'dog'", "nonroot"])
        .with_workdir("/testbed/")
    )
    for command in test_spec.env_script_list:
        ctr = ctr.with_exec(["bash", "-i", "-c", command])
    ctr = ctr.with_exec(
        [
            "bash",
            "-c",
            "echo 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed' >> /root/.bashrc",
        ]
    )
    for command in test_spec.repo_script_list:
        ctr = ctr.with_exec(["bash", "-i", "-c", command])

    return ctr


async def run_instance_dagger(
    instance: Instance,
    timeout: int,
    limiter: anyio.CapacityLimiter,
):
    """
    Run a single instance with the given prediction.

    Args:
        test_spec (TestSpec): TestSpec instance
        pred (dict): Prediction w/ model_name_or_path, model_patch, instance_id
        run_id (str): Run ID
        timeout (int): Timeout for running tests
    """
    async with limiter:
        logger = await instance.setup_logger()

        with anyio.move_on_after(timeout) as scope:
            ctr = get_instance_image(instance.test_spec)

            try:
                ctr = await _run_instance_patch(
                    instance,
                    ctr,
                    logger,
                )
                ctr = await _run_evaluation_script(instance, ctr, logger)

                # Get report from test output
                logger.info("Grading answer for %s...", instance.id)
                report = get_eval_report(
                    test_spec=instance.test_spec,
                    prediction=instance.pred,
                    test_log_path=str(instance.log_dir / TEST_LOG_FILE),
                    include_tests_status=True,
                )
                logger.info(
                    "report: %s\nResult for %s: resolved: %s",
                    report,
                    instance.id,
                    report[instance.id]["resolved"],
                )

                await anyio.Path(instance.log_dir / REPORT_FILE).write_text(
                    json.dumps(report, indent=4)
                )

            except EvaluationError as e:
                logger.info(str(e))

            except Exception:
                logger.exception(
                    "Error in evaluating module for %s.\nCheck %s for more information",
                    instance.id,
                    instance.log_dir / RUN_LOG_FILE,
                )

        if scope.cancel_called:
            logger.info("Evaluation for model %s timed out", instance.id)


async def _run_instance_patch(
    instance: Instance,
    ctr: dagger.Container,
    logger: Logger,
) -> dagger.Container:
    patch_diff = instance.pred.get("model_patch", "")
    await anyio.Path(instance.log_dir / PATCH_FILE).write_text(patch_diff)

    patch_file = f"/tmp/{PATCH_FILE}"
    ctr = ctr.with_new_file(patch_file, patch_diff)
    patched_ctr = ctr.with_exec(["git", "apply", "-v", patch_file])

    try:
        apply_patch_output = await patched_ctr.stdout()
    except dagger.ExecError:
        logger.info("Failed to apply patch to container, trying again...")
        patched_ctr = ctr.with_exec(
            [
                "patch",
                "--batch",
                "--fuzz=5",
                "-p1",
                "-i",
                patch_file,
            ],
        )
        try:
            apply_patch_output = await patched_ctr.stdout()
        except dagger.ExecError as e:
            msg = f"{APPLY_PATCH_FAIL}:\n{e.stderr}"
            if e.stdout:
                msg = f"{msg}\n\nStdout:\n{e.stdout}"
            raise EvaluationError(instance.id, msg, logger)

    logger.info(f"{APPLY_PATCH_PASS}:\n%s", apply_patch_output)

    return patched_ctr


async def _run_evaluation_script(
    instance: Instance,
    ctr: dagger.Container,
    logger: Logger,
) -> dagger.Container:
    # Get git diff before running eval script
    git_diff_output_before = await ctr.with_exec(
        ["git", "diff"],
    ).stdout()

    logger.info("Git diff before:\n%s", git_diff_output_before)

    test_output = ""
    for command in instance.test_spec.eval_script_list:
        # django hack
        command = command.replace("locale-gen", "locale-gen en_US.UTF-8")
        ctr = ctr.with_exec(["bash", "-i", "-c", command], expect=ReturnType.ANY)
        test_output += f"+ {command}\n"
        test_output += await ctr.stdout()
        if await ctr.exit_code() != 0:
            break

    await anyio.Path(instance.log_dir / TEST_LOG_FILE).write_text(test_output)

    logger.info(
        "Test output for %s written to %s",
        instance.id,
        instance.log_dir / TEST_LOG_FILE,
    )

    # Get git diff after running eval script
    git_diff_output_after = await ctr.with_exec(["git", "diff"]).stdout()

    # Check if git diff changed after running eval script
    logger.info("Git diff after:\n%s", git_diff_output_after)

    if git_diff_output_after != git_diff_output_before:
        logger.info("Git diff changed after running eval script")

    return ctr


def run_instances_dagger(
    predictions: dict,
    instances: list[SWEbenchInstance],
    full_dataset: list,
    run_id: str,
    max_workers: int,
    timeout: int,
):
    """
    Run all instances for the given predictions on Dagger.

    Args:
        predictions (dict): Predictions dict generated by the model
        instances (list): List of instances
        full_dataset (list):
        run_id (str): Run ID
        max_workers (int):
        timeout (int):
    """
    anyio.run(
        run_instances_dagger_async,
        predictions,
        instances,
        run_id,
        max_workers,
        timeout,
    )
    make_run_report(
        predictions,
        full_dataset,
        run_id,
    )


async def run_instances_dagger_async(
    predictions: dict,
    instances: list[SWEbenchInstance],
    run_id: str,
    max_workers: int,
    timeout: int,
):
    """
    Run all instances for the given predictions on Dagger.

    Args:
        predictions (dict): Predictions dict generated by the model
        instances (list): List of instances
        run_id (str): Run ID
        max_workers (int):
        timeout (int):
    """

    async def make_instance():
        for instance in instances:
            test_spec = await to_thread.run_sync(make_test_spec, instance)
            prediction = predictions[test_spec.instance_id]
            yield Instance(run_id, test_spec, prediction)

    limiter = anyio.CapacityLimiter(max_workers)

    cfg = dagger.Config()
    cfg.console.quiet = True

    async with dagger.connection(cfg), anyio.create_task_group() as tg:
        async for instance in make_instance():
            tg.start_soon(
                run_instance_dagger,
                instance,
                timeout,
                limiter,
            )
