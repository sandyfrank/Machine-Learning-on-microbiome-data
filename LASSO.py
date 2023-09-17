
# 1st install sklearn 
# import all the libraries neeeded 
import numpy as np
import pandas as pd
import gmtml 
import sys


def lasso(df_train, df_test, n_repeats , n_splits, nb_cpu):
    
    # Transforming our input data in to the format compatible with our code
    X_train, y_train =  gmtml.data_preparation(df_train)
    X_test, y_test =  gmtml.data_preparation(df_test)

    label_mapping = {1: 1 , -1 : 0} 
    y_trainEnc= np.vectorize(label_mapping.get)(y_train)
    y_testEnc= np.vectorize(label_mapping.get)(y_test)

    # Launching the RF
    lasso_best, lasso_cv_test_performance_df, lasso_cv_train_performance_df, lasso_FI_data, lasso_num_features_used,  mean_fprs_train, mean_fprs_test,mean_tprs_train, std_tpr_train, mean_aucs_train, std_aucs_train, mean_tprs_test, std_tpr_test, mean_aucs_test, std_aucs_test = gmtml.LASSO(Xtrain = X_train, ytrain = y_trainEnc, n_repeats = int(n_repeats), n_splits =  int(n_splits) , nb_cpu = int(nb_cpu))

    # Extracting the number of features with an importance different from zero
    lasso_num_features_used = pd.DataFrame({"lasso_num_features_used": lasso_num_features_used}, index=[0] )

    # Extract confusion metrix statistics on holdout validation data
    y_predprob = lasso_best.predict(X_test)
    y_pred = abs(y_predprob.round())

    tn, fp, fn, tp, roc_auc, accuracy, recall, precision, fscore, specificity, prevalence, fpr, tpr = gmtml.extract_confusion_matrix_stats(y = y_testEnc, y_pred = y_pred, y_predprob = y_predprob, lab = [0,1])  
    performance_metric = pd.DataFrame({'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'AUC': roc_auc, 'accuracy': accuracy,'recall': recall,'precision': precision,'f1_score': fscore,'specificity':specificity,'prevalence':prevalence}, index=['value'])

    # Exporting the outputs
    lasso_FI_data.to_csv('FI_data_LASSO.csv', index=False)
    lasso_cv_test_performance_df.to_csv('lasso_cv_test_performance_df.csv', index=False)
    lasso_cv_train_performance_df.to_csv('lasso_cv_train_performance_df.csv', index=False)
    performance_metric.to_csv('holdout_val_acc_metric_LASSO.csv', index=False)
    lasso_num_features_used.to_csv('lasso_num_features_used.csv', index = False)

    # Plotting the roc curve 
    gmtml.plot_cv_roc_curve(mean_fprs_train, mean_fprs_test, mean_tprs_train, mean_aucs_train, std_aucs_train, mean_tprs_test, mean_aucs_test, std_aucs_test, fpr, tpr, roc_auc, clf = 'lasso')


if __name__ == '__main__':
    df_train = sys.argv[1]
    df_test = sys.argv[2]
    n_repeats = sys.argv[3]
    n_splits = sys.argv[4]
    nb_cpu = sys.argv[5]

    lasso(df_train, df_test, n_repeats , n_splits, nb_cpu)



# Be in the folder containing your data
# python LASSO.py Bal_Data_cir_D.csv Bal_Data_cir_V.csv 1 2 20
# python LASSO.py Final_koat_train_data.crc.csv Final_koat_test_data.crc.csv 1 2 20





