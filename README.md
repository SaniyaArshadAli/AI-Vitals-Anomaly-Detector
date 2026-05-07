Prerequisites
You will need to install Kivy, Scikit-Learn, and Numpy.

pip install kivy scikit-learn numpy

Test it yourself:

Enter 80 for HR and 98 for SpO2. It will tell you it's normal.

Enter 110 for HR and 92 for SpO2. The ML model will instantly flag it as an anomaly because it falls outside the multi-dimensional cluster it learned during setup.
