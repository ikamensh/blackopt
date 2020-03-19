from blackopt.abc.solver import Solver, Problem, Solution
import math
import random


class SimAnneal(Solver):
    name = "Simulated Annealing"
    def __init__(self, problem: Problem, solution_cls, mutation_rate, temp = 0.5, half_times=20):

        super().__init__(problem, solution_cls)

        self.mutation_rate = mutation_rate
        self.temp_initial = temp
        self.temp = None
        self.half_times = half_times
        self.alpha = None
        self.current_solution = self.best_solution


    @property
    def cur_fitness(self):
        return self.current_solution.score

    @property
    def best_fitness(self):
        return self.best_solution.score


    def p_accept(self, candidate_fitness):
        """
        Probability of accepting if the candidate is worse than current
        Depends on the current temperature and difference between candidate and current
        """
        return math.exp(-abs(candidate_fitness - self.cur_fitness) / (self.problem.score_span * self.temp) )

    def maybe_accept(self, candidate: Solution):
        """
        Accept with probability 1 if candidate is better than current
        Accept with probabilty p_accept(..) if candidate is worse
        """
        candidate_fitness = candidate.score
        if candidate_fitness > self.cur_fitness:
            self.current_solution = candidate
            if candidate_fitness > self.best_fitness:
                self.best_solution = candidate

        else:
            if random.random() < self.p_accept(candidate_fitness):
                self.current_solution = candidate
            else:
                self.current_solution = self.best_solution

    def solve(self, steps):
        """
        Execute simulated annealing algorithm
        """
        self.problem.eval_count = 0
        self.temp = self.temp_initial
        self.alpha = 2 ** (- self.half_times / steps)
        doc_freq = 1 + steps // 500

        while self.problem.eval_count < steps:
            self.step()
            if not self.problem.eval_count % doc_freq :
                self.record()

    def step(self):
        candidate = self.current_solution.mutate(self.mutation_rate)
        self.maybe_accept(candidate)
        self.temp *= self.alpha

    def __str__(self):
        return f"{self.name} mr-{self.mutation_rate} ht-{self.half_times} t-{self.temp_initial}"


