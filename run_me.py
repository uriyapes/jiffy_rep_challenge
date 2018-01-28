import train_libras
import train_arabic_digits


def get_model(model_name):

    if model_name == "libras":
        model = train_libras.Model()
    elif model_name == "arabic digits":
        model = train_arabic_digits.Model()
    return model

def init_model(model):
    model.get_dataset()
    model.build_model()

def run_baseline(model):
    baseline_acc = model.run_baseline(model.dataset.train_set,
                                    model.dataset.train_labels,
                                    model.dataset.test_set,
                                    model.dataset.test_labels)
    # print('1NN Baseline accuarcy: %.3f' % baseline_acc)
    return baseline_acc

def run_model(model):
    model.train_model()
    test_set, test_embed_vec, test_labels, network_acc, nn_acc = model.eval_model()
    return network_acc, nn_acc


if __name__ == '__main__':
    model_name = "libras"
    l_baseline_acc = []
    num_of_model_runs = 2
    model = get_model(model_name)
    for i in xrange(num_of_model_runs):
        init_model(model)
        baseline_acc = run_baseline(model)
        network_acc, nn_acc = run_model(model)
        print('baseline accuracy: {0:.3f}\nnetwork accuracy: {1:.1f}%\n1-NN accuracy: {2:.3f}'.format(baseline_acc,network_acc,nn_acc))