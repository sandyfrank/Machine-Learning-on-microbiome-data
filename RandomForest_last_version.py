
# import all the libraries neeeded 
import pandas as pd
import GmtMl 
import sys


if __name__ == '__main__':
    df_train = sys.argv[1]
    df_test = sys.argv[2]
    n_repeats = sys.argv[3]
    n_splits = sys.argv[4]
    nb_cpu = sys.argv[5]

    runRF = GmtMl.Classifiers(df_train, df_test, n_repeats , n_splits, nb_cpu)
    output_dict = runRF.RandomForest()

    # Extracting the number of features with an importance different from zero
    rf_num_features_used = pd.DataFrame({"rf_num_features_used": output_dict['num_features_used']}, index=[0] )

    # Exporting the outputs
    output_dict['FI_data'].to_csv('rf_FI_data_RF.csv', index=False)
    output_dict['cv_test_performance_df'].to_csv('rf_cv_test_performance_df.csv', index=False)
    output_dict['cv_train_performance_df'].to_csv('rf_cv_train_performance_df.csv', index=False)
    output_dict['holdout_performance'].to_csv('holdout_performance_RF.csv', index=False)
    rf_num_features_used.to_csv('rf_num_features_used.csv', index = False)

    # Plotting the roc curve
    runRF.PlotRocCurves(output_dict, clf = 'rf')
    


# Be in the folder containing your data
# python RandomForest.py Bal_Data_cir_D.csv Bal_Data_cir_V.csv 10 10 20

# python RandomForest.py Final_koat_train_data.crc.csv Final_koat_test_data.crc.csv 10 10 20
