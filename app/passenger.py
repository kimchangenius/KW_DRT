class Passenger:
    def __init__(self, id, start, end, request_time, network):
        self.id = id
        self.start = start
        self.end = end
        self.request_time = request_time
        self.network = network
        self.pickup_time = None
        self.dropoff_time = None
        self.direct_route_time = network.get_travel_time(network.get_shortest_path(start, end))