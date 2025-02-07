# This file contains logic for running evaluations with Dagger: <https://dagger.io/>.

from __future__ import annotations

import asyncio
import json
import dagger
import modal.container_process
import modal.io_streams
import sys
import tenacity
import time
import traceback

from dataclasses import dataclass
from pathlib import Path
from swebench.harness.docker_build import setup_logger
from swebench.harness.reporting import make_run_report
from swebench.harness.utils import EvaluationError
from typing import cast

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    RUN_EVALUATION_LOG_DIR,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec


@dataclass
class TestOutput:
    instance_id: str
    test_output: str
    report_json_str: str
    run_instance_log: str
    patch_diff: str
    log_dir: Path
    errored: bool

def get_log_dir(pred: dict, run_id: str, instance_id: str) -> Path:
    model_name_or_path = cast(str, pred.get("model_name_or_path", "None").replace("/", "__"))
    return RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id

def get_instance_image(dag: dagger.Client, test_spec: TestSpec) -> dagger.Container:
    ctr = (
        dag.container().from_("ubuntu:22.04")
        .with_exec(["apt", "update"])
        .with_env_variable("DEBIAN_FRONTEND", "noninteractive")
        .with_env_variable("TZ", "Etc/UTC")
        .with_exec([
            "apt", "install", "-y",
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
            ])
        # .with_exec(["wget", "https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-2-Linux-x86_64.sh", "-O", "miniconda.sh"])
        .with_exec(["wget", "https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-2-Linux-aarch64.sh", "-O", "miniconda.sh"])
        .with_exec(["bash", "miniconda.sh", "-b", "-p", "/opt/miniconda3"])
        .with_exec(["bash", "-c", "echo 'export PATH=/opt/miniconda3/bin:$PATH' >> ~/.bashrc"])
        .with_exec(["/opt/miniconda3/bin/conda", "init", "--all"])
        .with_exec(["/opt/miniconda3/bin/conda", "config", "--append", "channels", "conda-forge"])
        .with_exec(["adduser", "--disabled-password", "--gecos", "'dog'", "nonroot"])
        .with_workdir("/testbed/")
    )
    for command in test_spec.env_script_list:
        ctr = ctr.with_exec(["bash", "-i", "-c", command])
    ctr = ctr.with_exec([
        "bash",
        "-c",
        "echo 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed' >> /root/.bashrc",
        ])
    for command in test_spec.repo_script_list:
        ctr = ctr.with_exec(["bash", "-i", "-c", command])

    return ctr

async def run_instance_dagger(
        dag: dagger.Connection,
        test_spec: TestSpec,
        pred: dict,
        run_id: str,
    ) -> TestOutput:
    """
    Run a single instance with the given prediction.

    Args:
        test_spec (TestSpec): TestSpec instance
        pred (dict): Prediction w/ model_name_or_path, model_patch, instance_id
        run_id (str): Run ID
        timeout (int): Timeout for running tests
    """
    instance_id = test_spec.instance_id
    log_dir = get_log_dir(pred, run_id, instance_id)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "run_instance.log"

    logger = setup_logger(instance_id, log_file, add_stdout=True)

    ctr = get_instance_image(dag, test_spec)

    patch_diff = pred.get("model_patch", "")
    try:
        patch_file = "/tmp/patch.diff"
        ctr = ctr.with_new_file(patch_file, patch_diff)
        patched_ctr = ctr.with_exec(["git", "apply", "-v", "/tmp/patch.diff"], expect=dagger.ReturnType.ANY)
        apply_patch_output = await patched_ctr.stdout()
        returncode = await patched_ctr.exit_code()

        if returncode != 0:
            logger.info(f"Failed to apply patch to container, trying again...")
            patched_ctr = ctr.with_exec(["patch", "--batch", "--fuzz=5", "-p1", "-i", "/tmp/patch.diff"], expect=dagger.ReturnType.ANY)
            returncode = await patched_ctr.exit_code()
            if returncode != 0:
                raise EvaluationError(
                    test_spec.instance_id,
                    f"{APPLY_PATCH_FAIL}:\n{patched_ctr.output}",
                    setup_logger(),
                )
            else:
                logger.info(f"{APPLY_PATCH_PASS}:\n{apply_patch_output}")
        else:
            logger.info(f"{APPLY_PATCH_PASS}:\n{apply_patch_output}")

        ctr = patched_ctr

        # Get git diff before running eval script
        git_diff_output_before = await ctr.with_exec(
            ["git", "diff"],
        ).stdout()
        logger.info(f"Git diff before:\n{git_diff_output_before}")

        # Run evaluation script
        test_output = ""
        for command in test_spec.eval_script_list:
            # django hack
            command = command.replace("locale-gen", "locale-gen en_US.UTF-8")
            ctr = ctr.with_exec(["bash", "-i", "-c", command])
            test_output += f"+ {command}\n"
            test_output += await ctr.stdout()
        test_output_path = log_dir / "test_output.txt"
        with open(test_output_path, "w") as f:
            f.write(test_output)
            logger.info(f"Test output for {instance_id} written to {test_output_path}")
            print(f"Test output for {instance_id} written to {test_output_path}")

        # Get git diff after running eval script
        git_diff_output_after = await ctr.with_exec(["git", "diff"]).stdout()

        # Check if git diff changed after running eval script
        logger.info(f"Git diff after:\n{git_diff_output_after}")
        if git_diff_output_after != git_diff_output_before:
            logger.info(f"Git diff changed after running eval script")

        # Get report from test output
        logger.info(f"Grading answer for {instance_id}...")
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        logger.info(
            f"report: {report}\n"
            f"Result for {instance_id}: resolved: {report[instance_id]['resolved']}"
        )
        return TestOutput(
            instance_id=instance_id,
            test_output=test_output,
            report_json_str=json.dumps(report, indent=4),
            run_instance_log=log_file.read_text(),
            patch_diff=patch_diff,
            log_dir=log_dir,
            errored=False,
        )
    except EvaluationError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        return TestOutput(
            instance_id=instance_id,
            test_output="",
            report_json_str="",
            run_instance_log=log_file.read_text(),
            patch_diff=patch_diff,
            log_dir=log_dir,
            errored=True,
        )
    except Exception as e:
        error_msg = (f"Error in evaluating model for {instance_id}: {e}\n"
                     f"{traceback.format_exc()}\n"
                     f"Check ({logger.log_file}) for more information.")
        logger.error(error_msg)
        return TestOutput(
            instance_id=instance_id,
            test_output="",
            report_json_str="",
            run_instance_log=log_file.read_text(),
            patch_diff=patch_diff,
            log_dir=log_dir,
            errored=True,
        )

def run_instances_dagger(
        predictions: dict,
        instances: list,
        full_dataset: list,
        run_id: str,
    ):
    """
    Run all instances for the given predictions on Dagger.

    Args:
        predictions (dict): Predictions dict generated by the model
        instances (list): List of instances
        run_id (str): Run ID
    """
    return asyncio.run(
        run_instances_dagger_async(predictions, instances, full_dataset, run_id)
    )

async def run_instances_dagger_async(
        predictions: dict,
        instances: list,
        full_dataset: list,
        run_id: str,
    ):
    """
    Run all instances for the given predictions on Dagger.

    Args:
        predictions (dict): Predictions dict generated by the model
        instances (list): List of instances
        run_id (str): Run ID
    """
    test_specs = list(map(make_test_spec, instances))

    # async with dagger.Connection(dagger.Config(log_output=sys.stdout)) as dag:
    async with dagger.Connection() as dag: # dagger.Config(log_output=sys.stdout)) as dag:
        # FIXME: sequential
        results = [
            await run_instance_dagger(
                dag,
                test_spec,
                predictions[test_spec.instance_id],
                run_id
            ) for test_spec in test_specs
        ]

        for result in results:
            result = cast(TestOutput, result)

            # Save logs locally
            log_dir = result.log_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            with open(log_dir / "run_instance.log", "w") as f:
                f.write(result.run_instance_log)
            with open(log_dir / "test_output.txt", "w") as f:
                f.write(result.test_output)
            with open(log_dir / "patch.diff", "w") as f:
                f.write(result.patch_diff)
            with open(log_dir / "report.json", "w") as f:
                try:
                    report_json = json.loads(result.report_json_str)
                    json.dump(report_json, f, indent=4)
                except Exception:
                    # This happens if the test fails with any exception
                    print(f"{result.instance_id}: no report.json")

    make_run_report(predictions, full_dataset, run_id)
