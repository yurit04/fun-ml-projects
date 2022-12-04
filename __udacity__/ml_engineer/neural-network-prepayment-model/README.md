# neural-network-prepayment-model

Here we build a neural-network based prepayment model to predict a likelihood of prepayments on pools of residential mortgage loans.

Mortgage pools are tradable fixed income securities composed of collections of residential mortgage loans. Here we only consider the so called "agency" mortgage pools, i.e. those whose credit risk (the risk of borrower's default) is guaranteed by one of the GSEs (Government Sponsored Enterprises) which are Fannie Mae, Freddie Mac, and Ginnie Mae. More specifically, we build a prepayment model here for predicting the likelyhood of residential mortgage prepayments for loans guaranteed by Fannie Mae. The same model can be used to predict prepayments on loans guaranteed by Freddie Mac, because they are very similar, but not Ginnie Mae loans, which are somewhat different and require a separate model. 

For more information about this project please, see **Proposal.pdf**, which contains more of the background information, problem statement, data description, and our approach to model building. 

## Files included

1. ### Proposal.pdf <br/>
      Describes the problem, gives background information, and outlines data structure and solution roadmap. 
1. ### pools_dataset_builder.ipynb  <br/>
      This can only be run by the author. It builds a dataset used in the model from raw data files, which the author has downloaded from a vendor. You won't be able to run this notebook  unless you have access to those files, but running this notebook is not at all required as all the data is provided in this project. The notebook, however, shows how the data is cleaned and NAs are filled. 
1. ### ./data  <br/>
      Directory which contains a series of csv files containing the data required to fit and avaluate the model. 
1. ### model_fitting.ipynb  <br/>
      This is the main notebook of the project. It is runnable. It contains all of the work done for data exploration, model fitting, and model evaluation.
1. ### ProjectReport.pdf  <br/>
      A project report required by the Udacity capstone project rubric. It contains the summary of the project and results. 
1. ### nn-ppm-from-csv.h5  <br/>
      If you want to avoid spending time fitting your own model while executing **model_fitting.ipynb**, see above, you can simply load the already fitted model from this file. 

## Core libraries used in this project

1. Pandas, Numpy
1. Matplotlib, Seaborn
1. Tensorflow, Keras
