from blackopt.algorithms import Gaos


def is_diverse(new, pop_sample, pressure):
    if len(pop_sample) == 0:
        return True
    avg_similarity = sum(p.similarity(new) for p in pop_sample) / len(pop_sample)
    return avg_similarity < 1 - pressure


class Rapga(Gaos):
    name = "Rapga"

    def solve(self, steps):

        while self.problem.eval_count < steps:

            next_generation = self.population[: self.elite_size]

            for i in range(30):
                pressure = 0.8 * self.problem.eval_count / steps
                new = self._breed(
                    self.popsize - self.elite_size,
                    pressure=pressure
                )
                next_generation += [c for c in new if is_diverse(new, next_generation, 0.3 + pressure/2)]

            self.population = next_generation

            self._rank()
            self.record()
            self.generation += 1
            if not self.generation % 5:
                print("Generation", self.generation, self.problem.eval_count)

        print(f"{self} is Done in {self.generation} generations")
