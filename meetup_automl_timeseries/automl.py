import numpy


class HyperParameter():
    def __init__(self, name, min_value, max_value, discrete = False):
        self.name = name
        self.mn = min_value
        self.mx = max_value
        self.discrete = discrete
        self.random_state = numpy.random.RandomState()

    def random_value(self):
        if self.discrete:
            value = self.random_state.randint(self.mn, self.mx+1)
        else:
            value = self.random_state.rand()*(self.mx - self.mn) + self.mn

        return value


class ConfigSpace(object):
    def __init__(self, classes):
        self.classes = {cls.__name__ : cls for cls in classes}

    def get_random_configuration(self):
        out_config = {}
        for name, cls in self.classes.items():
            out_config[name] = {}
            for hp in cls.get_Hyperparameters():
                out_config[name][hp.name] = hp.random_value()

        return out_config

    def run_configuration(self, x, config):
        additional_features = []
        for name in config:
            new_x = self.classes[name].run(x, **config[name])
            if new_x is not None:
                additional_features.append(new_x)

        print("%d new features were added" % len(additional_features))
        if len(additional_features)>0:
            new_x = numpy.concatenate(additional_features, axis=1)
            x = numpy.concatenate((x, new_x), axis=1)
        return x

#**** Example ****
class ConstantFeature():

    @classmethod
    def run(cls, x, n_lags=1, active=1):
        if active:
            return numpy.ones((x.shape[0], 1))*n_lags
        else:
            return None

    @classmethod
    def get_Hyperparameters(cls):
        active = HyperParameter(name='active', min_value=0, max_value=1, discrete=True)
        n_lags = HyperParameter(name='n_lags', min_value=3, max_value=6, discrete=True)
        return [n_lags, active]

config_space = ConfigSpace([ConstantFeature])

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