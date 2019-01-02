import numpy

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

from configurations import HyperParameter, ConfigSpace


#region estimators
class EstimatorBase():
    cls = None

    def __init__(self, **kwargs):
        self.estimator = self.cls(**kwargs)

    def fit(self, x, y):
        self.estimator.fit(x, y)

    def predict(self, x):
        return self.estimator.predict(x)

    @classmethod
    def get_hyperparameters(cls):
        raise NotImplementedError("abstract function")
    
    @classmethod
    def get_random_configuration(cls):
        out_config = {}

        for hp in cls.get_hyperparameters():
            out_config[hp.name] = hp.random_value()

        return out_config

    @classmethod
    def get_random_instance(cls):
        return cls(**cls.get_random_configuration())

class RandomForestWrapper(EstimatorBase):
    cls = RandomForestRegressor
    def __init__(self, **kwargs):
        if 'n_jobs' not in kwargs:
            kwargs['n_jobs'] = 3
        super(RandomForestWrapper, self).__init__(**kwargs)

    @classmethod
    def get_hyperparameters(cls):
        n_estimators = HyperParameter(name='n_estimators', min_value=10,
                                      max_value=100, discrete=True)
        max_depth = HyperParameter(name='max_depth', min_value=4,
                                   max_value=30, discrete=True)
        return [n_estimators, max_depth]


class ElasticNetWrapper(EstimatorBase):
    cls = ElasticNet

    def __init__(self, **kwargs):
        if 'tol' not in kwargs:
            kwargs['tol'] = numpy.power(10, kwargs['tol'])

        super(ElasticNetWrapper, self).__init__(**kwargs)

    @classmethod
    def get_hyperparameters(cls):
        alpha = HyperParameter(name='alpha', min_value=0, max_value=2, discrete=False)
        l1_ratio = HyperParameter(name='l1_ratio', min_value=0, max_value=1, discrete=False)
        tol = HyperParameter(name='tol', min_value=-4, max_value=-1,
                                  discrete=False)

        return [alpha, l1_ratio, tol]

#endregion


#region feature engineering classes
class ConstantFeature():

    @classmethod
    def run(cls, x, n_lags=1, active=1):
        if active:
            return numpy.ones((x.shape[0], 1))*n_lags
        else:
            return None

    @classmethod
    def get_hyperparameters(cls):
        active = HyperParameter(name='active', min_value=0, max_value=1, discrete=True)
        n_lags = HyperParameter(name='n_lags', min_value=3, max_value=6, discrete=True)
        window_size = HyperParameter(name='window_size', min_value=12, max_value=72, discrete=True)
        return [n_lags, active, window_size]


#endregion


if __name__ == '__main__':
    x = numpy.zeros((3, 1))
    print(x)

    config_space = ConfigSpace([ConstantFeature])
    config = config_space.get_random_configuration()
    print(config)
    x = config_space.run_configuration(x, config)
    print(x)
    config = config_space.get_random_configuration()
    print(config)
    x = config_space.run_configuration(x, config)
    print(x)
    config = config_space.get_random_configuration()
    print(config)
    x = config_space.run_configuration(x, config)
    print(x)