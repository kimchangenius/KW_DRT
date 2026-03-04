class RewardRecord:
    def __init__(self):
        self.reset()

    def reset(self):
        """에피소드 단위 보상 항목 초기화"""
        self.DecisionReward = 0.0
        self.ImmediateReward = 0.0
        self.DelayedReward = 0.0
        self.CancelPenalty = 0.0
        self.MaintenanceReward = 0.0
        return self

    def add_decision(self, value: float) -> float:
        self.DecisionReward += value
        return value

    def add_immediate(self, value: float) -> float:
        self.ImmediateReward += value
        return value

    def add_delayed(self, value: float) -> float:
        self.DelayedReward += value
        return value

    def add_cancel(self, value: float) -> float:
        self.CancelPenalty += value
        return value

    def add_maintenance(self, value: float) -> float:
        self.MaintenanceReward += value
        return value

    def total(self):
        return (
            self.DecisionReward
            + self.ImmediateReward
            + self.DelayedReward
            + self.CancelPenalty
            + self.MaintenanceReward
        )

    def as_dict(self):
        return {
            "decision": self.DecisionReward,
            "immediate": self.ImmediateReward,
            "delayed": self.DelayedReward,
            "cancel_penalty": self.CancelPenalty,
            "maintenance": self.MaintenanceReward,
            "total": self.total(),
        }
