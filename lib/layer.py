import tensorflow as tf

class GradientLayer(tf.keras.layers.Layer):
    """
    Custom layer to compute 1st and 2nd derivatives for Burgers' equation.

    Attributes:
        model: keras network model.
    """

    def __init__(self, model, **kwargs):
        """
        Args:
            model: keras network model.
        """

        self.model = model
        super().__init__(**kwargs)

    def call(self, x):
        """
        Computing 1st and 2nd derivatives for Burgers' equation.

        Args:
            x: input variable.

        Returns:
            model output, 1st and 2nd derivatives.
        """

        with tf.GradientTape() as g:
            g.watch(x)
            with tf.GradientTape() as gg:
                gg.watch(x)
                u = self.model(x)
            du_dtx = gg.batch_jacobian(u, x)
            du_dt = du_dtx[..., 0]
            du_dx = du_dtx[..., 1]
        d2u_dx2 = g.batch_jacobian(du_dx, x)[..., 1]
        return u, du_dt, du_dx, d2u_dx2
