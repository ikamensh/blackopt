from blackopt.examples.problems import TspSolution, TspProblem

def test_eval_count():

    prob = TspProblem.random_problem(5, 10)

    TspSolution.problem = prob
    sol = TspSolution.random_solution()

    score = sol.score
    score2 = sol.score

    assert prob.eval_count == 1