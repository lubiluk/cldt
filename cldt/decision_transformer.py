from cldt.policies import Policy


class DecisionTransformer(Policy):
    
    def __init__(self, env) -> None:
        super().__init__(env)
