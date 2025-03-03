import dataclasses
import multiprocessing
from typing import Annotated, Self

import dagger
from dagger import Doc, dag, function, object_type

MAX_WORKERS = round(multiprocessing.cpu_count() * 0.75)

@object_type
class SweBench:
    source: Annotated[
        dagger.Directory, 
        dagger.DefaultPath("/"),
        Doc("Source directory for the SWE-Bench project"),
    ]

    instances: list[str] = dataclasses.field(default_factory=list, init=False)
    logs: dagger.Directory | None = dataclasses.field(default=None, init=False)

    @function
    def build(self) -> dagger.Container:
        """Build the base container for running the evaluation"""
        ctr = (
            dag.container()
            .from_("ghcr.io/astral-sh/uv:python3.11-bookworm-slim")
            .with_mounted_cache("/root/.cache/uv", dag.cache_volume("uv-cache"))
            .with_env_variable("VIRTUAL_ENV", "/opt/venv")
            .with_exec(["uv", "venv", "$VIRTUAL_ENV"], expand=True)
            .with_env_variable("PATH", "$VIRTUAL_ENV/bin:$PATH", expand=True)
            .with_mounted_directory("/src", self.source)
            .with_workdir("/src")
            .with_exec(["uv", "pip", "install", "-e", "."])
            .with_workdir("/work")
        )
        if self.logs:
            ctr = ctr.with_directory("logs", self.logs)
        return ctr
    
    @function
    def with_instances(
        self,
        ids: Annotated[
            list[str],
            Doc("List of instance IDs to run"),
        ],
    ) -> Self:
        """Limit running with a set of instance IDs"""
        self.instances = ids
        return self        
    
    @function
    def with_logs(
        self, 
        directory: Annotated[
            dagger.Directory, 
            dagger.DefaultPath("/logs"),
            Doc("Directory with logs"),
        ],
    ) -> Self:
        """Add directory with existing logs to continue execution from"""
        self.logs = directory
        return self

    @function
    def run_evaluation(
        self, 
        run_id: Annotated[
            str,
            Doc("Identifies the run"),
        ] = "", 
        predictions_path: Annotated[
            str,
            Doc("Path to predictions file - if 'gold', uses gold predictions"),
         ] = "gold",
        max_workers: Annotated[
            int,
            Doc("Maximum number of workers (should be <= 75% of CPU cores)"),
        ] = MAX_WORKERS,
        timeout: Annotated[
            int | None,
            Doc("Timeout (in seconds) for running tests for each instance"),
        ] = None,
    ) -> dagger.Directory:
        """Run evaluation harness"""
        if not run_id:
            run_id = f"validate-{predictions_path}"
        args = [
            "python", 
            "-m", 
            "swebench.harness.run_evaluation", 
            "--run_id", 
            run_id, 
            "--predictions_path", 
            predictions_path,
            "--max_workers",
            str(max_workers),
            "--report_dir",
            "reports",
            "--dagger", 
            "true",
        ]
        if self.instances:
            args.extend(["--instance_ids", *self.instances])
        if timeout is not None:
            args.extend(["--timeout", str(timeout)])
        return (
            self.build()
            .with_exec(args, experimental_privileged_nesting=True)
            .directory("")
        )
