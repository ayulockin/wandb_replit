import numpy as np

def make_circle_dataset(n_samples: int = 100,
                        factor: float = 0.1,
                        noise: float = 0.05):
    """Credit: https://github.com/scikit-learn/scikit-learn/blob/80598905e517759b4696c74ecc35c6e2eb508cff/sklearn/datasets/_samples_generator.py#L641
  """
    # Number of samples in both the circles.
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # Generate the data
    linspace_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
    linspace_in = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)
    outer_circ_x = np.cos(linspace_out)
    outer_circ_y = np.sin(linspace_out)
    inner_circ_x = np.cos(linspace_in) * factor
    inner_circ_y = np.sin(linspace_in) * factor

    X = np.vstack([
        np.append(outer_circ_x, inner_circ_x),
        np.append(outer_circ_y, inner_circ_y)
    ]).T
    y = np.hstack([
        np.zeros(n_samples_out, dtype=np.intp),
        np.ones(n_samples_in, dtype=np.intp)
    ])

    generator = np.random.RandomState(42)
    X += generator.normal(scale=noise, size=X.shape)

    return X, np.expand_dims(y, axis=-1)