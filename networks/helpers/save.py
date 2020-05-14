import os
from json import dumps


def saveResults(network, epochs, denses, results):
    result = {
        "network": network,
        "epochs": epochs,
        "denses": denses,
        "results": results
    }

    experiments = [int(k.split('experiment')[-1].split('.json')[0])
                   for k in os.listdir('./results')]
    experiments.sort()

    with open(f'./results/experiment{experiments[-1]+1}.json', 'w') as file:
        file.write(dumps(result))
