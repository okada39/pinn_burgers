import tensorflow as tf
from .layer import GradientLayer

class PINN:
    """
    Build a physics informed neural network (PINN) model for Burgers' equation.

    Attributes:
        network: keras network model with input (t, x) and output u(t, x).
        nu: kinematic viscosity.
        grads: gradient layer.
    """

    def __init__(self, network, nu):
        """
        Args:
            network: keras network model with input (t, x) and output u(t, x).
            nu: kinematic viscosity.
        """

        self.network = network
        self.nu = nu
        self.grads = GradientLayer(self.network)

    def build(self):
        """
        Build a PINN model for Burgers' equation.

        Returns:
            PINN model for the projectile motion with
                input: [ (t, x) relative to equation,
                         (t=0, x) relative to initial condition,
                         (t, x=bounds) relative to boundary condition ],
                output: [ u(t,x) relative to equation (must be zero),
                          u(t=0, x) relative to initial condition,
                          u(t, x=bounds) relative to boundary condition ]
        """

        # equation input: (t, x)
        tx_eqn = tf.keras.layers.Input(shape=(2,))
        # initial condition input: (t=0, x)
        tx_ini = tf.keras.layers.Input(shape=(2,))
        # boundary condition input: (t, x=-1) or (t, x=+1)
        tx_bnd = tf.keras.layers.Input(shape=(2,))

        # compute gradients
        u, du_dt, du_dx, d2u_dx2 = self.grads(tx_eqn)

        # equation output being zero
        u_eqn = du_dt + u*du_dx - self.nu*d2u_dx2
        # initial condition output
        u_ini = self.network(tx_ini)
        # boundary condition output
        u_bnd = self.network(tx_bnd)

        # build the PINN model for Burgers' equation
        return tf.keras.models.Model(
            inputs=[tx_eqn, tx_ini, tx_bnd], outputs=[u_eqn, u_ini, u_bnd])
