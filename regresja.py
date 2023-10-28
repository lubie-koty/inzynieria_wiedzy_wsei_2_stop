import matplotlib.pyplot as plt
import numpy
import pandas


if __name__ == '__main__':
    csv_data = pandas.read_csv('headbrain.csv')
    head_size_values = csv_data['Head Size(cm^3)'].values
    head_size_mean = numpy.mean(head_size_values)
    head_size_max = numpy.max(head_size_values) + 100
    head_size_min = numpy.min(head_size_values) - 100
    brain_weight_values = csv_data['Brain Weight(grams)'].values
    brain_weight_mean = numpy.mean(brain_weight_values)

    numerator = 0
    denominator = 0
    for i in range(len(head_size_values)):
        numerator += (head_size_values[i] - head_size_mean) * (brain_weight_values[i] - brain_weight_mean)
        denominator += (head_size_values[i] - head_size_mean) ** 2

    a = numerator / denominator
    b = brain_weight_mean - (a * head_size_mean)
    print(f'{a=:.3f}\n{b=:.3f}')

    x = numpy.linspace(head_size_min, head_size_max, 1000)
    y = b + a * x
    plt.rcParams['figure.figsize'] = (20.0, 10.0)
    plt.plot(x, y, color='#C06C84', label='Regresja liniowa')
    plt.scatter(head_size_values, brain_weight_values, color='#355C7D', label='Dane pomiarowe')
    plt.xlabel('Objetosc czaszki [cm\u00B3]')
    plt.ylabel('Waga mozgu [g]')
    plt.legend()
    plt.show()

    ss_t = 0
    ss_r = 0
    for i in range(len(head_size_values)):
        y_pred = b + a * head_size_values[i]
        ss_t += (brain_weight_values[i] - brain_weight_mean) ** 2
        ss_r += (brain_weight_values[i] - y_pred) ** 2
    r_2 = 1 - (ss_r / ss_t)
    print(f'{r_2=:.4f}')
