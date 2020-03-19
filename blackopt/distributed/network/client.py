import pickle
import socket
from typing import Tuple, Any, Type

from boltons.socketutils import NetstringSocket

from blackopt.abc import Problem
from blackopt.distributed.network.config import host, port



class ClientSocket:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.name = None
        self.next_order = 1
        self.orders_stored = {}
        self.ns_socket = NetstringSocket(self.socket)

    def connect(self) -> Tuple[Problem, Type]:
        self.socket.connect((host, port))
        greeting = self.ns_socket.read_ns().decode("utf-8", "strict")
        print(f"Greeted by server: {greeting}")

        transmission = self.ns_socket.read_ns()
        prob, solution_cls = pickle.loads(transmission)
        return prob, solution_cls

    def send(self, best, some):
        data_string = pickle.dumps((best, some))
        self.ns_socket.write_ns(data_string)

    def receive(self) -> Any:
        data_string = self.ns_socket.read_ns()
        new = pickle.loads(data_string)
        return new
