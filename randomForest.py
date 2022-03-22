from util import rmse, get_dataset_sklearn, plot_results, load_sklearn_model
from sklearn.ensemble import RandomForestRegressor
import pickle

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset args
    parser.add_argument('-t','--training', help='Train the model', 
                        action="store_true")
    # Model Evaluation args
    parser.add_argument('-e','--evaluation', help='Eval model', 
                        action="store_true")

    args = parser.parse_args()

    if args.training:
        for day in range(96):
            X, y = get_dataset_sklearn(day=day, farm=0, type_data="train", gap=48, 
                                       history_size=96, forecast_horizon=96)
            print("X", X.shape)
            print("y", y.shape)

            rfr = RandomForestRegressor(n_estimators=50, criterion='squared_error', 
                                        max_depth=None, min_samples_split=2, 
                                        min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                                        max_features='auto', max_leaf_nodes=None, 
                                        min_impurity_decrease=0.0, bootstrap=True, 
                                        oob_score=False, n_jobs=4, random_state=None, 
                                        verbose=0, warm_start=False, 
                                        ccp_alpha=0.0, max_samples=None)

            if args.training:
                rfr = rfr.fit(X, y)

                # save
                with open(f'model/RandomForest/rfr_{day}.pkl','wb') as f:
                    pickle.dump(rfr, f)

            else:
                with open(f'model/RandomForest/rfr_{day}.pkl', 'rb') as f:
                    rfr = pickle.load(f)

            y_hat = plot_results(model=rfr, 
                                 X=X, y=y, 
                                 save_path="results/RandomForest/{day}.png", 
                                 sklearn=True, 
                                 show=False)

        print("rmse:", rmse(y_hat, y), "rmse normalized", rmse(y_hat, y)/30_000)


    if args.evaluation:
        fh = 1
        day = 0
        X, y = get_dataset_sklearn(day=day, farm=0, type_data="train", gap=48, 
                                   history_size=96, forecast_horizon=96)
        num_samples_train = X.shape[0]
        Y_train = np.empty((num_samples_train, fh))
        Y_valid = np.empty((num_samples_train, fh))
        
        rfr = load_sklearn_model(path_to_model=f"model/RandomForest/rfr_{day}.pkl")        

        Y.append(rfr.predict(X))



