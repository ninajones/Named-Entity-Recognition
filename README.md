# Named-Entity-Recognition

## About
Named Entity Recognition involves identifying portions of text representing labels such as geographical location, geopolitical entity, persons, etc. In this tutorial, we will use deep learning to identify various entities in Medium articles and present them in useful way.

## Dependencies
jupyter=1.0.0

jupyter_client=6.1.2
jupyter_console=6.1.0
jupyter_core=4.6.3
keras=2.1.6
keras-applications=1.0.8
keras-preprocessing=1.1.0
matplotlib=3.2.1
matplotlib-base=3.2.1
nltk==3.4.5
numpy==1.18.2
pandas=1.0.3
pickle-mixin==1.0.2
python=3.6.10
scikit-learn=0.22.2.post1
scipy==1.4.1
sklearn==0.0
tensorboard==1.8.0
tensorflow==1.8.0
tensorflow-estimator==2.1.0
tensorflow-hub==0.3.0

## How to Use
Option #1:
Run Train_NER_Model.ipynb in a directory with this dataset. Save the outputs (the model weights and generated tags) in the same directory as Train_NER_Model.ipynb. Then run NER.ipynb in the same directory.

Option #2:
Run Train_NER_Model.ipynb in Jupyter Notebook. Create a new project in an IDE such as PyCharm. Save this dataset, main.py, clean.py, entity_recognition.py, and the outputs from Train_NER_Model.ipynb in the same project directory. Run main.py.
