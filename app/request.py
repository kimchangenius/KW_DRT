class Request:
    def __init__(self, user_id, from_node_id, to_node_id, request_time):
        self.user_id = user_id
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.request_time = request_time
        self.travel_time = -10000000

    def __str__(self):
        return (f"Request(user_id={self.user_id}, "
                f"from={self.from_node_id}, to={self.to_node_id}, "
                f"request_time={self.request_time}, travel_time={self.travel_time})")

    def __repr__(self):
        return (f"Request({self.user_id}, {self.from_node_id}, "
                f"{self.to_node_id}, {self.request_time}, {self.travel_time})")

