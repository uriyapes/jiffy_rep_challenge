import csv
import numpy as np

import jiffy_model


def get_model(model_name, embed_vec_size):
    model = jiffy_model.Model(model_name, embed_vec_size)
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

def print_results_lists(l_baseline_acc, l_network_acc, l_nn_acc):
    print "baseline accuracy list:"
    print l_baseline_acc
    print "\nnetwork accuracy list:"
    print l_network_acc
    print "\n1-NN accuracy list: "
    print l_nn_acc


def write_results_to_csv(list, file_name = 'results.csv'):
    with open(file_name, 'ab') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ')  #DELETE: quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(list)

def run_model_multiple_times(model_name, num_of_model_runs, embed_vec_size = 40, run_baseline_flag = True):
    l_baseline_acc = []
    l_network_acc = []
    l_nn_acc = []
    model = get_model(model_name, embed_vec_size)
    for i in xrange(num_of_model_runs):
        print('########## Number of model run: {0} ##########'.format(i))
        init_model(model)
        if run_baseline_flag:
            baseline_acc = run_baseline(model)
            l_baseline_acc.append(baseline_acc)
            print('baseline accuracy: {0:.3f}'.format(baseline_acc))
        network_acc, nn_acc = run_model(model)
        l_network_acc.append(network_acc)
        l_nn_acc.append(nn_acc)
        print('network accuracy: {0:.3f}%\n1-NN accuracy: {1:.3f}'.format(network_acc,nn_acc))

    print_results_lists(l_baseline_acc, l_network_acc, l_nn_acc)
    return l_baseline_acc, l_network_acc, l_nn_acc


def run_model_with_diff_hyperparams(model_name, num_of_model_runs_per_config, embed_size_l, run_baseline_flag = False):
    file_name_template = r"results/results_embed_size_"
    l_avg_nn_acc = []
    l_avg_network_acc = []
    for i in xrange(len(embed_size_l)):
        file_name = file_name_template + str(embed_size_l[i])
        l_baseline_acc, l_network_acc, l_nn_acc = run_model_multiple_times\
                                                (model_name, num_of_model_runs_per_config, embed_size_l[i], run_baseline_flag)

        if run_baseline_flag:
            write_results_to_csv(l_baseline_acc, file_name)
        write_results_to_csv(l_network_acc, file_name)
        write_results_to_csv(l_nn_acc, file_name)
        l_avg_network_acc.append(np.mean(l_network_acc))
        l_avg_nn_acc.append(np.mean(l_nn_acc))

    file_name = r"results/avg_results_over_" + str(num_of_model_runs_per_config) + "_runs"
    write_results_to_csv(embed_size_l, file_name)
    write_results_to_csv(l_avg_network_acc, file_name)
    write_results_to_csv(l_avg_nn_acc, file_name)


if __name__ == '__main__':
    model_name = "arabic digits"
    # model_name = "libras"
    num_of_model_runs = 1
    run_baseline_flag = False

    embedded_size_list = [30, 40, 50]
    # embedded_size_list = [40]
    run_model_with_diff_hyperparams(model_name, num_of_model_runs, embedded_size_list, run_baseline_flag)

    # run_model_multiple_times(model_name, num_of_model_runs)