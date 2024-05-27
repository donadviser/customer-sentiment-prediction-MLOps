# Predicting how a customer will feel about a product before they even ordered it
## Pre-Purchase Sentiment Prediction with MLOps


**Problem statement**: For a given customer's historical data, we are tasked to
predict the review score for the next order or purchase. 

We will be using the [The Kaggle Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). This dataset has information on 100,000 orders from 2016 to 2018 made at
multiple marketplaces in Brazil. Its features allow viewing charges from various
dimensions: from order status, price, payment, freight performance to customer
location, product attributes and finally, reviews written by customers. The
objective here is to predict the customer satisfaction score for a given order
based on features like order status, price, payment, etc. 

In order to achieve this in a real-world scenario, we will be using [ZenML](https://zenml.io/) to
build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.


## Install Packages
- create virtual environment: I use pyenv
  `pyenv virtualenv 3.11 venv311_zenml`
  `pyenv local venv311_zenml`
  
`pip install "zenml["server"]"`

`zenml up`

To integrate zenml with mflow, run the following command
`zenml integration install mlflow -y`

The project can only be executed with a ZenML stack that has an MLflow
experiment tracker and model deployer as a component. Configuring a new stack
with the two components are as follows:

```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register local-mlflow-stack -a default -o default -d mlflow -e mlflow_tracker --set
```

This should give you the following stack to work with. 

![mlflow_stack](_assets/mlflow_stack.png)


Add this line before callint the training pipeline
`print(Client().active_stack.experiment_tracker.get_tracking_uri())`

use the output to as parameter below:
`mlflow ui --backend-store-uri "file:/Users/don/Library/Application Support/zenml/local_stores/afe9b1d8-06da-4d66-bcd1-d75224ab95f4/mlruns"`


