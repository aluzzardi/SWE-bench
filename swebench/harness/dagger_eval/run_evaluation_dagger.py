# This file contains logic for running evaluations with Dagger: <https://dagger.io/>.

from __future__ import annotations

from collections import Counter
import functools
import json
import logging
import time
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import cast
import sys

import anyio
import dagger
from anyio import to_thread
from dagger import ReturnType, dag
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import get_tracer_provider
from opentelemetry import trace
from . import telemetry


from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    RUN_EVALUATION_LOG_DIR,
)
from swebench.harness.constants.constants import KEY_INSTANCE_ID, SWEbenchInstance
from swebench.harness.docker_build import close_logger, setup_logger
from swebench.harness.grading import get_eval_report
from swebench.harness.reporting import make_run_report
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec
from swebench.harness.utils import EvaluationError

RUN_LOG_FILE = "run_instance.log"
TEST_LOG_FILE = "test_output.txt"
PATCH_FILE = "patch.diff"
REPORT_FILE = "report.json"


logging.getLogger("httpx").setLevel(logging.ERROR)

tracer = telemetry.get_tracer()


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


async def run_instance_dagger(
    bench_instance: SWEbenchInstance,
    prediction: dict[str, str],
    run_id: str,
    timeout: int,
    limiter: anyio.CapacityLimiter,
):
    """
    Run a single instance with the given prediction.

    Args:
        bench_instance (SWEbenchInstance): SWE-bench instance
        pred (dict): Prediction w/ model_name_or_path, model_patch, instance_id
        run_id (str): Run ID
        timeout (int): Timeout for running tests
    """
    async with limiter:
        test_spec = await to_thread.run_sync(make_test_spec, bench_instance)
        instance = Instance(run_id, test_spec, prediction)
        logger = await instance.setup_logger()

        try:
            with tracer.start_as_current_span(instance.id) as span:
                with anyio.move_on_after(timeout) as scope:
                    ctr = await get_instance_image(instance.test_spec)
                    ctr = await _run_instance_patch(instance, ctr, logger)
                    ctr = await _run_evaluation_script(instance, ctr, logger)
                    resolved = await _run_report(instance, logger)

                    if not resolved:
                        span.set_status(trace.StatusCode.ERROR, "not resolved")

                if scope.cancel_called:
                    logger.info("Evaluation for model %s timed out", instance.id)

        except EvaluationError as e:
            logger.info(str(e))

        except Exception:
            logger.exception(
                "Error in evaluating module for %s.\nCheck %s for more information",
                instance.id,
                instance.log_dir / RUN_LOG_FILE,
            )

        finally:
            # TODO: write log file here instead of leaving it open while this is running
            # to avoid too many open files
            close_logger(logger)


@tracer.start_as_current_span("setup base container")
async def get_instance_image(test_spec: TestSpec) -> dagger.Container:
    return await (
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
                "python3.11",
            ]
        )
        .with_exec(
            [
                "update-alternatives",
                "--install",
                "/usr/bin/python",
                "python",
                "/usr/bin/python3.11",
                "7",
            ]
        )
        .with_file(
            "miniconda.sh",
            dag.http(
                "https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-2-Linux-aarch64.sh"
            ),
        )
        .with_exec(["bash", "miniconda.sh", "-b", "-p", "/opt/miniconda3"])
        .with_env_variable("PATH", "/opt/miniconda3/bin:$PATH", expand=True)
        .with_exec(["conda", "init", "--all"])
        .with_exec(["conda", "config", "--append", "channels", "conda-forge"])
        .with_exec(["adduser", "--disabled-password", "--gecos", "'dog'", "nonroot"])
        .with_new_file(
            "/root/setup_env.sh",
            test_spec.setup_env_script,
            permissions=0o755,
        )
        .with_new_file("/root/setup_repo.sh", test_spec.install_repo_script)
        .with_exec(["/root/setup_env.sh"])
        .with_exec(
            [
                "bash",
                "-c",
                "echo 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed' >> /root/.bashrc",
            ]
        )
        .with_exec(["bash", "/root/setup_repo.sh"])
        .with_workdir("/testbed")
        .sync()
    )


@tracer.start_as_current_span("apply patch")
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
        apply_patch_output = await patched_ctr.stderr()
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
            apply_patch_output = await patched_ctr.stderr()
        except dagger.ExecError as e:
            msg = f"{APPLY_PATCH_FAIL}:\n{e.stderr}"
            if e.stdout:
                msg = f"{msg}\n\nStdout:\n{e.stdout}"
            raise EvaluationError(instance.id, msg, logger)

    logger.info(f"{APPLY_PATCH_PASS}:\n%s", apply_patch_output)

    return patched_ctr


@tracer.start_as_current_span("run evaluation script")
async def _run_evaluation_script(
    instance: Instance,
    ctr: dagger.Container,
    logger: Logger,
) -> dagger.Container:
    # Get git diff before running eval script
    git_diff_output_before = await ctr.with_exec(["git", "diff"]).stdout()

    logger.info("Git diff before:\n%s", git_diff_output_before)

    failures = Counter()

    start_time = time.time()
    test_output = ""
    for command in instance.test_spec.eval_script_list:
        if command in (
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
        ):
            continue

        # django hack
        command = command.replace("locale-gen", "locale-gen en_US.UTF-8")

        with tracer.start_as_current_span(command) as span:
            ctr = ctr.with_exec(
                ["bash", "--login", "-c", command],
                redirect_stdout="/out",
                redirect_stderr="/out",
                expect=ReturnType.ANY,
            )
            test_output += f"+ {command}\n"
            test_output += await ctr.file("/out").contents()
            code = await ctr.exit_code()

            failures[bool(code)] += 1

            if code:
                error = trace.StatusCode.ERROR
                span.set_status(error, f"command failed with exit code {code}")

    total_runtime = time.time() - start_time
    logger.info(f"Test runtime: {total_runtime:_.2f} seconds")

    await anyio.Path(instance.log_dir / TEST_LOG_FILE).write_text(test_output)

    if failures[True]:
        trace.get_current_span().set_status(
            trace.StatusCode.ERROR,
            f"{failures[True]} command(s) failed (out of {failures.total()})",
        )

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


@tracer.start_as_current_span("generate report")
async def _run_report(instance: Instance, logger: Logger):
    # Get report from test output
    logger.info("Grading answer for %s...", instance.id)

    report = await to_thread.run_sync(
        functools.partial(
            get_eval_report,
            test_spec=instance.test_spec,
            prediction=instance.pred,
            test_log_path=str(instance.log_dir / TEST_LOG_FILE),
            include_tests_status=True,
        )
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

    return report[instance.id]["resolved"]


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
    cast(TracerProvider, get_tracer_provider()).shutdown()


async def run_instances_dagger_async(
    predictions: dict[str, dict[str, str]],
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

    limiter = anyio.CapacityLimiter(max_workers)

    cfg = dagger.Config()
    cfg.log_output = sys.stderr
    cfg.console.quiet = True

    async with dagger.connection(cfg), anyio.create_task_group() as tg:
        for instance in instances:
            tg.start_soon(
                run_instance_dagger,
                instance,
                predictions[instance[KEY_INSTANCE_ID]],
                run_id,
                timeout,
                limiter,
            )
