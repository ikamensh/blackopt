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

