# multilingualFedDep

This is the repository for our paper "Federated learning for privacy-preserving depression detection with multilingual language models in social media posts". 
It explores the potential of applying federated learning on multilingual depressive data. The used publicly available data and the codes producing the results found in the paper are attached. Unfortunately, two of the used datasets are not open to the public, which means that the results cannot be repeated.


The three publicly available datasets, Arabic, Russian, and Korean, are uploaded with the distribution 
used while developing our model. Only 10k records were used from the Russian dataset.
The python file data_proc_utils in the data directory contains the functions used to process the data.
English and Spanish datasets are not publicly available but can be obtained by contacting their providers.
The five datasets sources:
1. English: https://www.cs.jhu.edu/~mdredze/clpsych-2015-shared-task-evaluation/ 
2. Arabic: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YHMYEQ
3. Russian: https://data.mendeley.com/datasets/838dbcjpxb/1
4. Spanish: https://www.jmir.org/2019/6/e14199/ 
5. Korean: https://github.com/dxlabskku/Mental-Health/tree/main/data (twitter_Korean.csv)

The two files required for creating the two environments are "environment_fed.txt" for the federated learning environment 
and "environment_cnt_lcl.yml" for making the centralized/local learning environment. 

To run the code, you only need to set the parameters in each yaml file and run the corresponding python file. 
For example, if I want to train a centralized or a local model on specific data, I would change the parameters in "cnt_lcl_train_config.yaml" save it, then run "cnt_lcl_train.py". An empty folder titled "results" in the code's directory must be created for the model to be saved.
