'''
a loss function quantifies how good or bad a
given predictor (scoring function) is at classifying the input data
points in a dataset.
'''

import numpy as np
import pandas as pd

predicted_scores = {
	"Dog": [4.26, 1.33, -1.01],
	"Cat": [3.76, -1.20, -3.81],
	"Panda": [-2.37, 1.03, -2.27]
}

scoring_func_result = pd.DataFrame(data = predicted_scores, index = ["Pred_Dog", "Pred_Cat", "Pred_Panda"])
print(scoring_func_result)

scoring_arr = scoring_func_result.values
print(scoring_arr)

# hinge loss

L_1 = max(0, 1.33 - 4.26 + 1) + max(0, -1.01 - 4.26 + 1)
L_2 = max(0, 3.76 - (-1.20) + 1) + max(0, -3.81 - (-1.20) + 1)
L_3 = max(0, -2.37 - (-2.27) + 1) + max(0, 1.03 - (-2.27) + 1)

# a data point is correctly classified when L = 0
print(L_1)
print(L_2)
print(L_3)

L = (L_1 + L_2 + L_3) / 3.0
print(L)





