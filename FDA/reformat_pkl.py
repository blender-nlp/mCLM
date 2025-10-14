# On machine with NumPy 2.1.3
import pickle
import numpy as np


for task in ['ames', 'bbbp', 'cyp3a4', 'dili', 'hia', 'pgp']:

    with open(f"saved_improve_{task}.pkl", "rb") as f:
        data = pickle.load(f)

    converted_data = data

    converted_data['scores'] = [l.tolist() for l in converted_data['scores'][task][0]]

    #print(converted_data['scores'])
    #zz

    with open(f"saved_improve_{task}.new.pkl", "wb") as f:
        pickle.dump(converted_data, f)


