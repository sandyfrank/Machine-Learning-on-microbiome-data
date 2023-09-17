"""Module containing all the classification algorithms and a predicting function

   X data should have samples in rows and features in columns

   y data should be a vector with length equals to number of samples of X, with levels control - case

   X data should be filtered, with groups balanced before using here

   Make sure that the row names of X and y are the same

"""

"""
    You need to install the following depences
    sklearn, xgboost
    data = [X|y]
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, confusion_matrix
from joblib import dump, load
import sklearn.ensemble as skle
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb
import sys

############################# OOP FORMAT #########################################"

class CommonMethods :

    def __init__(self, df_train, df_test, n_repeats, n_splits, ncpu):

        self.y = None
        self.y_pred = None
        self.y_predprob = None
        self.lab = None
        self.cv_output = None
        self.holdout_output=None
        self.clf= None
        self.df_train = df_train
        self.df_test = df_test
        self.nb_cpu = int(ncpu)
        self.n_repeats = int(n_repeats)
        self.n_splits = int(n_splits)
        self.mean_fprs_train = np.linspace(0, 1, 100)
        self.mean_fprs_test = np.linspace(0, 1, 100)

    def clean_data(self):

        """
        This method separates the data into a label vector y and the remainder X.
        It encodes y into {1,-1}

        """

        self.df_train = pd.read_csv(self.df_train)
        self.df_test = pd.read_csv(self.df_test)

        # Naming indexes
        self.df_train.index = self.df_train.iloc[:, 0]
        self.df_test.index = self.df_test.iloc[:, 0]

        # Dropping a useless column

        self.df_train = self.df_train.drop(['Unnamed: 0'], axis=1)
        self.df_test = self.df_test.drop(['Unnamed: 0'], axis=1)

        # Separating into species abundances table (X) and the outcomes (y) training and testing
        
        ytrain, Xtrain  = self.df_train['y'], self.df_train.drop(['y'], axis=1)
        ytest, Xtest  = self.df_test['y'], self.df_test.drop(['y'], axis=1)

        # Encode the labels by creating a dictionary
        label_mapping = {'control': 1, 'case': -1}
        ytrainEnc = np.array(ytrain.map(label_mapping))
        ytestEnc = np.array(ytest.map(label_mapping))

        return Xtrain, ytrainEnc, Xtest, ytestEnc



    def extract_confusion_matrix_stats(self):

        """
        This method uses predicted and true labels to calculate the confusion matrix
        and extract performance measures (recall, precision, specificity, accuracy, prevalence, f-score).

        """

        fpr, tpr, thresholds = roc_curve(self.y, self.y_predprob)
        roc_auc = auc(fpr, tpr)

        # Compute the confusion matrix with positive label = 1
        conf_mat = confusion_matrix(self.y, self.y_pred, labels= self.lab)

        # Extract the true negatives, true positives, false negatives, and false positives
        tn, fp, fn, tp = conf_mat.ravel()

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        specificity = tn / (tn + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        prevalence = (tp + fn) / (tp + tn + fp + fn)
        fscore = 2 * precision * recall / (recall + precision)

        output_vec = pd.DataFrame({'tn':[tn], 'fp':[fp], 'fn':[fn], 'tp':[tp], 'roc_auc':[roc_auc],
                           'accuracy':[accuracy], 'recall':[recall], 'precision':[precision], 
                           'fscore':[fscore],'specificity':[specificity], 'prevalence' : [prevalence],
                           'fpr':[fpr], 'tpr':[tpr]})
        # Assign row names
        output_vec.index = ['value']

        return (output_vec)
    
   
    
    def PlotRocCurves(self, cv_output, clf):

        """
        To plot the ruc curves of the mean cv train, mean cv test and of the holdout validation 

        """
        plt.plot(cv_output['mean_fprs_train'], cv_output['mean_tprs_train'], color='darkorange',
                  lw=2, label='Mean cv_train ROC curve (AUC = %0.2f $\pm$ %0.2f)' % (cv_output['mean_aucs_train'],
                                                                                      cv_output['std_aucs_train']))
        
        plt.plot(cv_output['mean_fprs_test'], cv_output['mean_tprs_test'], color='r',
                  lw=2, label='Mean cv_test ROC curve (AUC = %0.2f $\pm$ %0.2f)' % (cv_output['mean_aucs_test'],
                                                                                      cv_output['std_aucs_test']))       
        plt.plot(cv_output['holdout_performance']['fpr'][0] , cv_output['holdout_performance']['tpr'][0] , color='b', 
                 label = 'Holdout val. ROC Curve (AUC = {:.2f})'.format(cv_output['holdout_performance']['roc_auc'].item()))
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.savefig(str(clf) + '_roc_curves.pdf')  # Save the plot as a PDF file
        plt.show()



class Classifiers(CommonMethods) :

    def __init__(self, df_train, df_test, n_repeats, n_splits, ncpu):
        super().__init__(df_train, df_test, n_repeats, n_splits, ncpu)  
    
    def Evaluation(self, Xtrain, ytrain, best_model):

        """
        This method uses the best model resulting from hyperparameter tuning to perform repeated cross-validation. 
        The user decides on the number of folds and the number of cross-validation repetitions.
        Performance measures for the test and training sets of the cross-validation are displayed in separate dataframes.

        """

        cv_train_performance = []
        cv_test_performance = []

        for seed in range(self.n_repeats):
            skf = StratifiedKFold(n_splits=self.n_splits, random_state=seed, shuffle=True)

            tprs_train, roc_aucs_train = [], []
            tprs_test, roc_aucs_test = [], []

            for fold, (train, test) in enumerate(skf.split(Xtrain, ytrain)):


                X_train, X_test = Xtrain.iloc[train], Xtrain.iloc[test]
                y_train, y_test = ytrain[train], ytrain[test]

                best_model.fit(X_train, y_train)
                y_predprob_train = best_model.predict_proba(X_train)[:, 1]
                y_pred_train = best_model.predict(X_train)

                # Modifying values 
                self.y = y_train
                self.y_pred =  y_pred_train
                self.y_predprob = y_predprob_train
                self.lab = [-1,1]

                extrain = self.extract_confusion_matrix_stats()

                tprs_train.append(np.interp(self.mean_fprs_train, extrain.loc['value','fpr'], extrain.loc['value','tpr']))
                tprs_train[-1][0] = 0.0
                roc_aucs_train.append(auc(extrain.loc['value','fpr'], extrain.loc['value','tpr']))

                cv_train_performance.append({ 
                    'seed': seed + 1, 'Fold': fold + 1, 'tn_train': extrain.loc['value','tn'], 
                    'fp_train': extrain.loc['value','fp'],'fn_train': extrain.loc['value','fn'], 
                    'tp_train': extrain.loc['value','tp'], 'AUC_train': extrain.loc['value','roc_auc'],
                    'accuracy_train':extrain.loc['value','accuracy'], 'recall_train': extrain.loc['value','recall'],
                    'precision_train': extrain.loc['value','precision'], 'f1_score_train':extrain.loc['value','fscore'],
                    'specificity_train': extrain.loc['value','specificity'], 'prevalence_train': extrain.loc['value','prevalence']
                })

                y_predprob_test = best_model.predict_proba(X_test)[:, 1]
                y_pred_test = best_model.predict(X_test)

                # Modifying values 
                self.y = y_test
                self.y_pred =  y_pred_test
                self.y_predprob = y_predprob_test
                self.lab = [-1,1]

                extest = self.extract_confusion_matrix_stats()

                # Add the result to the list
                tprs_test.append(np.interp(self.mean_fprs_test, extest.loc['value','fpr'],extest.loc['value','tpr']))
                tprs_test[-1][0] = 0.0
                roc_aucs_test.append(auc(extest.loc['value','fpr'], extest.loc['value','tpr']))

                
                cv_test_performance.append({ 
                    'seed': seed + 1, 'Fold': fold + 1, 'tn_test': extest.loc['value','tn'], 
                    'fp_test': extest.loc['value','fp'],'fn_test': extest.loc['value','fn'], 
                    'tp_test': extest.loc['value','tp'], 'AUC_test': extest.loc['value','roc_auc'],
                    'accuracy_test':extest.loc['value','accuracy'], 'recall_test': extest.loc['value','recall'],
                    'precision_test': extest.loc['value','precision'], 'f1_score_test':extest.loc['value','fscore'],
                    'specificity_test': extest.loc['value','specificity'], 'prevalence_test': extest.loc['value','prevalence']
                })


        return tprs_train, roc_aucs_train, cv_train_performance, tprs_test, roc_aucs_test, cv_test_performance
 

    def RandomForest(self):

        """
        This is the main random forest method. It is sequential:
        - Calls the clean_data method to separate the data into X and y.
        - Performs a hyperparameter tuning to obtain the best model and feature importances.
        - Calls up the Evaluation method that evaluates the best model following a cross-validation scheme.
        - Evaluate the generalization of the best model on the holdout validation data by calling the extract_confusion_matrix_stats method.

        Result:
        A dictionary of intermediate results data frames.
        
        """

        Xtrain, ytrain, Xtest, ytest = self.clean_data()

        # Train the Random Forest classifier
        rfc = skle.RandomForestClassifier(random_state=42)
        rfc.fit(Xtrain, ytrain)

        # Setting the parameter grid for HP tuning
        param_grid = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10, 15],
            'max_features': [10, 15, 19, 50, 369]
        }

        # Create a GridSearchCV object with 10-fold stratified cross-validation
        cv = StratifiedKFold(n_splits=self.n_splits)
        grid_search = GridSearchCV(
            rfc, param_grid=param_grid, cv = cv, scoring= 'roc_auc',
            return_train_score= True, n_jobs=self.nb_cpu
        )

        # Fit the GridSearchCV object to the data
        grid_search.fit(Xtrain, ytrain)

        # Print the cv results for all the splits
        results = pd.DataFrame(grid_search.cv_results_)

        # Get the best Random Forest model from the grid search
        rf_best = grid_search.best_estimator_

        # Get features importance and keep in a DataFrame
        feature_importances = rf_best.feature_importances_

        # Number of features used
        num_features_used = len([i for i in feature_importances if i > 0])

        # Sort the features by importance in descending order
        sorted_indices = np.argsort(feature_importances)[::-1]
        sorted_importances = feature_importances[sorted_indices]
        sorted_features = Xtrain.columns[sorted_indices]

        FI = list(zip(sorted_features, sorted_importances))
        FI_data = pd.DataFrame(FI, columns=['features', 'importances'])  # to be returned

        Xtrain.rename(index={idx: i for i, idx in enumerate(Xtrain.index)}, inplace=True)

        tprs_train, roc_aucs_train, cv_train_performance, tprs_test, roc_aucs_test, cv_test_performance = self.Evaluation( Xtrain, ytrain, best_model = rf_best)

        # Extract confusion metrix statistics on holdout validation data
        y_predprob_holdout = rf_best.predict_proba(Xtest)[:, 1]
        y_pred_holdout = rf_best.predict(Xtest) 

        # Modifying values 
        self.y = ytest
        self.y_pred =  y_pred_holdout
        self.y_predprob = y_predprob_holdout
        self.lab = [-1,1]

        exholdout = self.extract_confusion_matrix_stats()


        # Compute the mean and std FPR and TPR across all folds for train and test sets
        mean_tprs_train, std_tpr_train = np.mean(tprs_train, axis=0), np.std(tprs_train, axis=0)
        mean_aucs_train, std_aucs_train = np.mean(roc_aucs_train), np.std(roc_aucs_train)

        mean_tprs_test, std_tpr_test = np.mean(tprs_test, axis=0), np.std(tprs_test, axis=0)
        mean_aucs_test, std_aucs_test = np.mean(roc_aucs_test), np.std(roc_aucs_test)

        # Convert the results list to a DataFrame
        cv_test_performance_df = pd.DataFrame(cv_test_performance)  # to be returned
        cv_train_performance_df = pd.DataFrame(cv_train_performance)  # to be returned

        # Train the Random Forest model with best hyperparameters
        rf_best.fit(Xtrain, ytrain)  # to be returned

        output_dict = {
            'rf_best': rf_best, 'cv_test_performance_df': cv_test_performance_df, 'cv_train_performance_df':cv_train_performance_df,
            'FI_data':FI_data, 'num_features_used': num_features_used, 'mean_fprs_train': self.mean_fprs_train, 'holdout_performance':exholdout,
            'mean_fprs_test':self.mean_fprs_test,'mean_tprs_train':mean_tprs_train, 'std_tpr_train':std_tpr_train, 
            'mean_aucs_train':mean_aucs_train, 'std_aucs_train':std_aucs_train,'mean_tprs_test':mean_tprs_test, 
            'std_tpr_test':std_tpr_test, 'mean_aucs_test':mean_aucs_test, 'std_aucs_test':std_aucs_test

        }

        return (output_dict)
    

    
    






    
        

        