import law
import luigi

from FLAF.run_tools.law_customizations import Task, HTCondorWorkflow, copy_param


class HelloWorldTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 0.1)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)
    poll_interval = copy_param(HTCondorWorkflow.poll_interval, 1)
    bundle_flavours = ["core"]
    force_fail = luigi.BoolParameter(
        default=False,
        significant=False,
        description="raise an exception in run() to test log transfer on crash",
    )

    def create_branch_map(self):
        return {0: "hello"}

    def output(self):
        return self.remote_target(
            self.version, self.__class__.__name__, self.period, "hello_world_done.txt"
        )

    def run(self):
        if self.force_fail:
            raise RuntimeError("Forced failure for testing log transfer on crash")
        print("hello world")
        with self.output().localize("w") as tmp:
            with open(tmp.path, "w") as f:
                f.write("done\n")
