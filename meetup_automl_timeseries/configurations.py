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
            for hp in cls.get_hyperparameters():
                out_config[name][hp.name] = hp.random_value()

        return out_config

    def run_configuration(self, x, config):
        additional_features = []
        for name in config:
            new_x = self.classes[name].run(x, **config[name])
            if new_x is not None:
                additional_features.append(new_x)

        if len(additional_features)>0:
            new_x = numpy.concatenate(additional_features, axis=1)
            x = numpy.concatenate((x, new_x), axis=1)
        return x
