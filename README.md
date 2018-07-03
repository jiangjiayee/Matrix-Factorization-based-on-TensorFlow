# Matrix-Factorization-based-on-TensorFlow
Matrix Factorization based on TensorFlow with both Explicit and Implicit information


# Use exam.py to test the model
In exam.py, the parameter implicit control the explicit or implicit information
<br>Case True:it will do the implicit information,and you will get the AUC score,which is 0.888 of ml-1m and 0.904 of ml-latest-small
<br>Case False:it will do the explicit ratings,and you will get the MSE score,which is 0.844 of ml-1m and 0.811 of ml-latest-small
