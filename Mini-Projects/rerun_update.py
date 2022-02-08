# Make sure that we use SageMaker 1.x
!pip install sagemaker==1.72.0

import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import pandas as pd
import new_data

new_X, new_Y = new_data.get_new_data()
new_vectorizer = CountVectorizer(max_features=5000,
                preprocessor=lambda x: x, tokenizer=lambda x: x)
new_vectorizer.fit(new_X)
new_XV = new_vectorizer.transform(new_X).toarray()
new_val_X = pd.DataFrame(new_XV[:10000])
new_train_X = pd.DataFrame(new_XV[10000:])

new_val_y = pd.DataFrame(new_Y[:10000])
new_train_y = pd.DataFrame(new_Y[10000:])
pd.DataFrame(new_XV).to_csv(os.path.join(data_dir, 'new_data.csv'), header=False, index=False)
pd.concat([new_val_y, new_val_X], axis=1).to_csv(os.path.join(data_dir, 'new_validation.csv'), header=False, index=False)
pd.concat([new_train_y, new_train_X], axis=1).to_csv(os.path.join(data_dir, 'new_train.csv'), header=False, index=False)

data_dir = '../data/sentiment_update'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

import sagemaker

session = sagemaker.Session() # Store the current SageMaker session

# S3 prefix (which folder will we use)
prefix = 'sentiment-update'

from sagemaker import get_execution_role

# Our current execution role is require when creating the model as the training
# and inference code will need to access the model artifacts.
role = get_execution_role()

from sagemaker.amazon.amazon_estimator import get_image_uri

container = get_image_uri(session.boto_region_name, 'xgboost')

xgb = sagemaker.estimator.Estimator(container, # The location of the container we wish to use
                                    role,                                    # What is our current IAM Role
                                    train_instance_count=1,                  # How many compute instances
                                    train_instance_type='ml.m4.xlarge',      # What kind of compute instances
                                    output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                    sagemaker_session=session)

xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        silent=0,
                        objective='binary:logistic',
                        early_stopping_rounds=10,
                        num_round=500)

new_data_location = session.upload_data(os.path.join(data_dir, 'new_data.csv'), key_prefix=prefix)
new_val_location = session.upload_data(os.path.join(data_dir, 'new_validation.csv'), key_prefix=prefix)
new_train_location = session.upload_data(os.path.join(data_dir, 'new_train.csv'), key_prefix=prefix)


s3_input_train = sagemaker.s3_input(s3_data=train_location, content_type='csv')
s3_input_validation = sagemaker.s3_input(s3_data=val_location, content_type='csv')

xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})

new_xgb.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

