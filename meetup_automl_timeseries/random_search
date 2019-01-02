## Evaluation Code (Do not edit)
import numpy

from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
from meetup_automl_timeseries.configurations import ConfigSpace

# no need to change this function
from meetup_automl_timeseries.classes import EstimatorBase, \
    RandomForestWrapper, ConstantFeature, ElasticNetWrapper


def evaluate(X, y, preproc_config_space: ConfigSpace, preproc_config : dict, estimator: EstimatorBase, validation_size=1000,
             test_size=1000):

    X = preproc_config_space.run_configuration(X, preproc_config)

    Xtrain = X[:-validation_size - test_size]
    Ytrain = y[:-validation_size - test_size]
    Xval = X[-validation_size - test_size:-test_size]
    Yval = y[-validation_size - test_size:-test_size]
    Xtest = X[-test_size:]
    Ytest = y[-test_size:]

    estimator.fit(Xtrain, Ytrain)

    test_predictions = estimator.predict(Xtest)
    val_predictions = estimator.predict(Xval)

    test_loss = mean_absolute_error(y_true=Ytest, y_pred=test_predictions)
    val_loss = mean_absolute_error(y_true=Yval, y_pred=val_predictions)
    return val_loss, test_loss


# no need to change this function
def run_experiments(X, y, n_experiments, estimator_wrapper: EstimatorBase, preproc_config_space: ConfigSpace):

    val_losses = []
    test_losses = []
    try_index = []
    best_val_loss = 1e100
    best_config = (None, None)

    for i in range(n_experiments):
        preproc_config = preproc_config_space.get_random_configuration()
        estimator_kwargs = estimator_wrapper.get_random_configuration()
        estimator = estimator_wrapper(**estimator_kwargs)

        val_loss, test_loss = evaluate(X, y,
                                       preproc_config_space=preproc_config_space,
                                       preproc_config=preproc_config,
                                       estimator=estimator,
                                       validation_size=500,
                                       test_size=500)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            val_losses.append(best_val_loss)
            test_losses.append(test_loss)
            try_index.append(i)
            best_config = (preproc_config, estimator_kwargs)
            print("experiment %d: Validation loss = %.3g, test loss = %.3g" % (i, val_loss, test_loss))

    print("Best validation loss = %.3g, test loss = %.3g" % (val_losses[-1], test_losses[-1]))
    print("Best config:\n%s\n%s" % best_config)

    plt.plot(try_index, val_losses, '.-b', try_index, test_losses, '.-r')
    plt.show()

if __name__ == '__main__':

    X = numpy.random.rand(3000,10)
    y = numpy.sum(X, axis=1) + 3*numpy.std(X, axis=1)

    run_experiments(X, y, n_experiments=500,
                    estimator_wrapper=ElasticNetWrapper,
                    preproc_config_space=ConfigSpace([ConstantFeature]))