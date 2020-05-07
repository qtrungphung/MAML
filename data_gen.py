import numpy as np


class WaveFunction:
    """ Create a wave function of the form
            f(x) = A*sin(x + p)
            A: amplitude
            p: phase
        
        Method:
            __init__ receive A and p as parameters
            __call__ receive a list of x's. Return np.ndarray of f(x) at x's
    """

    def __init__(self, A, p):
        assert isinstance(A, float)
        assert isinstance(p, float)
        self.A = A
        self.p = p

    def __call__(self, x):
        assert isinstance(x, (list, np.ndarray, torch.Tensor))
        return self.A * np.sin(x + self.p)


def task_sampler(num_tasks: int):
    """Generate a sine wave regression task. Specifically:  
        - Generate amplitude (A) and phase (p) of a sine wave  

    Args:  
        num_tasks: number of tasks  

    Return:  
        tuple (As, ps)
        As: list of num_tasks Amplitudes
        ps: list of num_tasks phases
    """

    assert isinstance(num_tasks, int)
    A_range = np.arange(0.1, 0.5, 0.001)
    p_range = np.arange(0.0, np.pi, 0.001)
    A = np.random.choice(A_range, size=(num_tasks,))
    p = np.random.choice(p_range, size=(num_tasks,))

    return A, p


def gen_tasks(num_tasks: int, num_samples: int = 10):
    """ Generate a number of tasks

    Args:
        num_tasks: number of task to generate
        num_samples: K=10 as in paper [1]
    """

    x_range = np.arange(-5.0, 5.0, 0.001)  # as in paper [1]
    As, ps = task_sampler(num_tasks)
    wave_funcs = [WaveFunction(A, p) for A, p in zip(As, ps)]
    data = []
    for wave_func in wave_funcs:
        xs = np.random.choice(x_range, size=(num_samples, 1))
        ys = wave_func(xs)
        yield xs, ys, wave_func
