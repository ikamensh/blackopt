from blackopt.distributed.network.server import ServerSocket

from blackopt.examples.problems import TspProblem, TspSolution
prob = TspProblem.random_problem(2, 90)

server = ServerSocket(prob, TspSolution)
server.listen()



