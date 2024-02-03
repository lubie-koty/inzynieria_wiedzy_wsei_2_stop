import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import skfuzzy.control as ctrl


if __name__ == '__main__':
    quality = ctrl.Antecedent(np.arange(0, 11, 1), 'quality')
    service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
    tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

    quality.automf(3)
    service.automf(3)

    tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
    tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
    tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])

    quality['average'].view()
    service.view()
    tip.view()
    plt.show()

    rule_1 = ctrl.Rule(quality['poor'] | service['poor'], tip['low'])
    rule_2 = ctrl.Rule(service['average'], tip['medium'])
    rule_3 = ctrl.Rule(service['good'] | quality['good'], tip['high'])

    rule_1.view()
    plt.show()

    tipping_ctrl = ctrl.ControlSystem([rule_1, rule_2, rule_3])
    tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

    tipping.input['quality'] = 6.5
    tipping.input['service'] = 9.8

    tipping.compute()
    print(tipping.output['tip'])
