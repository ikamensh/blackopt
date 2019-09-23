from blackbox.abc.solver import Solver

class RandomSearch(Solver):
    name = "random search"

    def solve(self, n_evaluations):

        doc_freq = 1 + n_evaluations // 500

        for i in range(n_evaluations):
            solution = self.solution_cls.random_solution()
            if solution.score > self.best_solution.score:
                self.best_solution = solution

            if not i % doc_freq:
                print(i)
                self.record()

