# multilingualFedDep

This is the repository for our unpublished paper "Towards Privacy-Conscious Multilingual Mental Health Analysis: A Federated Learning Perspective on Depression Detection". 
It explores the potential of applying federated learning on multilingual depressive data. The used publicly available data and the codes producing the results found in the paper are attached. Unfortunately, two of the used datasets are not open to the public, which means that the results cannot be repeated.

To run the code, you only need to set the parameters in each yaml file and run the corresponding python file. 
For example, if I want to train a centralized or a local model on specific data, I would change the parameters in "cnt_lcl_train_config.yaml" save it, then run "cnt_lcl_train.py"
