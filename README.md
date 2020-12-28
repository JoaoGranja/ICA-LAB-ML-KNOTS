# ICA-LAB-ML-KNOTS

## Overview

ICA-LAB-ML-KNOTS is a research project of [ICA] (https://www.ica-canyoning.org) association to study the feasibility to apply deep learning models on canyoning knots images classification. The project is written in Python and the main goal is to provide a web page capable classifying input user images with canyoning knots. 


### Contents

* [Project](#project)
  * [Installation](#project-installation)
  * [Usage](#project-usage)

## Project

### Project installation

(Optional first step)

```
conda create -n ica-lab-ml-knots python=3.7
conda activate ica-lab-ml-knots
```
...
```{bash}
git clone https://github.com/JoaoGranja/ICA-LAB-ML-KNOTS.git
cd dodo/PyDodo
pip install .
```

### Project usage

If BlueBird (and a simulator) are running, then one can communicate with BlueBird using PyDodo. For example:

 ```python
 >>> streamlit run main.py
 ```
