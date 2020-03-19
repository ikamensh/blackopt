from blackopt.abc.solver import Solver

class RandomSearch(Solver):
    name = "random search"

    def solve(self, steps):

        doc_freq = 1 + steps // 500

        for i in range(steps):
            solution = self.solution_cls.random_solution()
            if solution.score > self.best_solution.score:
                self.best_solution = solution

            if not i % doc_freq:
                print(i)
                self.record()

    def step(self):
        solution = self.solution_cls.random_solution()
        if solution.score > self.best_solution.score:
            self.best_solution = solution


import pathos
import dill

class MulticoreRS(Solver):
    name = "multicore random search"

    def solve(self, steps):
        self.problem.eval_count = 0
        pool = pathos.pools.ProcessPool()
        n_cpus = pool.ncpus

        mapping_steps = int(steps ** (1 / 5))
        self.steps_per_pool = steps // (n_cpus * mapping_steps)

        print(self.steps_per_pool)
        print(mapping_steps)
        print(n_cpus)

        to_solve = dill.copy(self)

        for i in range(mapping_steps):
            solutions = pool.map(self._solve_pool, [to_solve] * n_cpus)
            self.best_solution = max(
                (s for s in solutions + [self.best_solution]), key=lambda x: x.score
            )
            self.problem.eval_count += self.steps_per_pool * n_cpus
            self.record()
            print(i * n_cpus * self.steps_per_pool)

    @staticmethod
    def _solve_pool(self: 'MulticoreRS'):

        doc_freq = 1 + self.steps_per_pool // 500

        for i in range(self.steps_per_pool):
            solution = self.solution_cls.random_solution()
            if solution.score > self.best_solution.score:
                self.best_solution = solution

            if not i % doc_freq:
                self.record()

        return self.best_solution
