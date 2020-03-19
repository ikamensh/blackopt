import pickle
import socket
from threading import Thread
import typing
import random

from boltons.socketutils import NetstringSocket

from blackopt.distributed.network.config import host, port


class ServerSocket:
    def __init__(self, prob, solution_cls):

        self.prob = prob
        self.solutions_cls = solution_cls
        self.solutions_set = set()

        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serversocket.bind((host, port))
        self.serversocket.listen(5)

        self.active_clients: typing.Dict[NetstringSocket, str] = {}

    def listen(self):
        print(f"{self} is listening.")
        while True:
            (sock, address) = self.serversocket.accept()
            sock = NetstringSocket(sock)
            print(f"new connection: {address}!")

            self.active_clients[sock] = address
            sock.write_ns(bytes("Welcome to endless optimization service.", encoding="utf-8"))
            sock.write_ns( pickle.dumps( (self.prob, self.solutions_cls) ) )

            ct = Thread(target=ServerSocket.client_thread, args=(self, sock))
            ct.start()

    def client_thread(self, clientsocket: NetstringSocket):
        while True:
            data = clientsocket.read_ns()
            best, some = pickle.loads(data)
            self.solutions_set.add(best)
            self.solutions_set.add(some)

            to_send = pickle.dumps(random.sample(self.solutions_set, 1)[0])
            clientsocket.write_ns(to_send)
