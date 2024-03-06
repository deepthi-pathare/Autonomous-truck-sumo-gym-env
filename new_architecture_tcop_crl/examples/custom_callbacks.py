from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        action = self.training_env.get_attr('action',indices=[0])[0]
        lvd = self.training_env.get_attr('lvd',indices=[0])[0]
        speeds = self.training_env.get_attr('speeds',indices=[0])[0]
        self.logger.record("Action", action)
        self.logger.record("Lead Vehicle Distance", lvd)
        self.logger.record("Ego Speed", speeds[0,0])
        return True
    
class EntropyDecayCallback(BaseCallback):
    """
    Custom callback for adaptively changing the entropy coefficient hyperparameter of the model 
    (applicable for PPO and A2C).
    """

    def __init__(self, ent_coef_initial, ent_coef_final, decay_steps, verbose=0):
        super(EntropyDecayCallback, self).__init__(verbose)
        self.ent_coef_initial = ent_coef_initial
        self.ent_coef_final = ent_coef_final
        self.decay_steps = decay_steps

    def _on_step(self):
        current_step = self.num_timesteps
        if current_step <= self.decay_steps:
            # Calculate the new entropy coefficient
            ent_coef = self.ent_coef_initial - (((self.ent_coef_initial - self.ent_coef_final) / self.decay_steps) * current_step)

            # Set the new entropy coefficient
            self.model.ent_coef = ent_coef  
        return True