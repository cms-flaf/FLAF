import law

from FLAF.run_tools.law_customizations import Task, HTCondorWorkflow, copy_param


class HelloWorldTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 1.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)

    def create_branch_map(self):
        return {0: "hello"}

    def output(self):
        return self.local_target("hello_world_done.txt")

    def run(self):
        print("hello world")
        with open(self.output().path, "w") as f:
            f.write("done\n")
