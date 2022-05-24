from util import rmse, get_dataset_sklearn, plot_results, load_sklearn_model, get_dataset_rnn, simple_plot
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
import pickle
import os

import argparse


if __name__ == '__main__':
    models = {"RandomForestRegressor": RandomForestRegressor,
              "ExtraTreesRegressor": ExtraTreesRegressor,
              "LinearRegression": LinearRegression,
              "Ridge": Ridge}


    parser = argparse.ArgumentParser()
    # Dataset args
    parser.add_argument('-t','--training', help='Train the model', 
                        action="store_true")

    parser.add_argument('-c_t','--continue_training',
        help='Continue the training. Requires the path number of the last forest trained', 
        required=False, default=0, type=int)
    # Model Evaluation args
    parser.add_argument('-e','--evaluation', help='Eval model', 
                        action="store_true")

    parser.add_argument('--dataset_size', help='Eval model', 
                        type=str, default="small")

    parser.add_argument('--model', help='Eval model', 
                        required=True,
                        type=str, default="RandomForestRegressor", 
                        choices=models.keys())

    args = parser.parse_args()

    model = models[args.model]

    start = args.continue_training

    if not os.path.isdir(f'model/{args.model}'):
        os.mkdir(f'model/{args.model}')

    if args.training:
        for quarter in range(start, 96):
            X, y = get_dataset_sklearn(quarter=quarter, farm=0, type_data="train", gap=48, 
                                       history_size=96, forecast_horizon=96, size=args.dataset_size)
            if quarter % 12 == 0:
                print("X", X.shape)
                print("y", y.shape)

            if model == LinearRegression or model == Ridge:
                rfr = model()
            else:
                rfr = model(n_estimators=50, criterion='squared_error', 
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
                with open(f'model/{args.model}/{args.model}_{quarter}.pkl','wb') as f:
                    pickle.dump(rfr, f)

            else:
                with open(f'model/{args.model}/{args.model}_{quarter}.pkl', 'rb') as f:
                    rfr = pickle.load(f)


    if args.evaluation:
        fh = 96
        quarter = 0

        X_train, y_train = get_dataset_sklearn(quarter=quarter, farm=0, type_data="train", gap=48, 
                                   history_size=96, forecast_horizon=96, size=args.dataset_size)
        X_valid, y_valid = get_dataset_sklearn(quarter=quarter, farm=0, type_data="valid", gap=48, 
                                   history_size=96, forecast_horizon=96, size=args.dataset_size)
        X_test, y_test = get_dataset_sklearn(quarter=quarter, farm=0, type_data="test", gap=48, 
                                   history_size=96, forecast_horizon=96, size=args.dataset_size)

        num_samples_train = X_train.shape[0]
        num_samples_valid = X_valid.shape[0]
        num_samples_test = X_test.shape[0]

        Y_train = np.empty((num_samples_train, fh))
        Y_valid = np.empty((num_samples_valid, fh))
        Y_test = np.empty((num_samples_test, fh))

        Y_train_truth = np.empty((num_samples_train, fh))
        Y_valid_truth = np.empty((num_samples_valid, fh))
        Y_test_truth = np.empty((num_samples_test, fh))


        for quarter in range(fh):
            print("quarter", quarter)
            X_train, y_train = get_dataset_sklearn(quarter=quarter, farm=0, type_data="train", gap=48, 
                                       history_size=96, forecast_horizon=96, size=args.dataset_size)
            X_valid, y_valid = get_dataset_sklearn(quarter=quarter, farm=0, type_data="valid", gap=48, 
                                       history_size=96, forecast_horizon=96, size=args.dataset_size)
            X_test, y_test = get_dataset_sklearn(quarter=quarter, farm=0, type_data="test", gap=48, 
                                   history_size=96, forecast_horizon=96, size=args.dataset_size)
            
            print("X_train", X_train.shape)
            rfr = load_sklearn_model(path_to_model=f"model/{args.model}/{args.model}_{quarter}.pkl")        

            Y_train[:,quarter] = rfr.predict(X_train)
            Y_valid[:,quarter] = rfr.predict(X_valid)
            Y_test[:,quarter] = rfr.predict(X_test)

            Y_train_truth[:,quarter] = y_train
            Y_valid_truth[:,quarter] = y_valid
            Y_test_truth[:,quarter] = y_test


        y_train = Y_train_truth
        y_valid = Y_valid_truth
        y_test = Y_test_truth

        path_save_image = f"results/figure/{args.model}/"
        if not os.path.isdir(path_save_image):
            os.mkdir(path_save_image)

        losses_train = np.sqrt(np.mean(np.square(Y_train - y_train[:,:fh]), axis=1)) # [rmse(Y_train[i,:],y_train[i,:fh]) for i in range(num_samples_train)] # np.sqrt(np.mean(np.square(Y_train - y_train[:,:fh]), axis=1))
        simple_plot(truth=y_train[0,:fh], forecast=Y_train[0], periods=fh, save=path_save_image + f"{args.model}_train.png")
        print(f"rmse: {np.mean(losses_train):.4f} \pm {np.std(losses_train):.4f}")

        losses_valid = np.sqrt(np.mean(np.square(Y_valid - y_valid[:,:fh]), axis=1)) # [rmse(Y_valid[i,:],y_valid[i,:fh]) for i in range(num_samples_valid)]# np.sqrt(np.mean(np.square(Y_valid - y_valid[:,:fh]), axis=1))
        simple_plot(truth=y_valid[0,:fh], forecast=Y_valid[0], periods=fh, save=path_save_image + f"{args.model}_valid.png")
        print(f"rmse: {np.mean(losses_valid):.4f} \pm {np.std(losses_valid):.4f}")

        losses_test = np.sqrt(np.mean(np.square(Y_test - y_test[:,:fh]), axis=1)) # [rmse(Y_valid[i,:],y_valid[i,:fh]) for i in range(num_samples_valid)]# np.sqrt(np.mean(np.square(Y_valid - y_valid[:,:fh]), axis=1))
        simple_plot(truth=y_valid[0,:fh], forecast=Y_test[0], periods=fh, save=path_save_image + f"{args.model}_test.png")
        print(f"rmse: {np.mean(losses_test):.4f} \pm {np.std(losses_test):.4f}")

        best = np.argmin(losses_train)
        print(f"Best rmse: {losses_train[best]}")
        simple_plot(truth=y_train[best,:fh], forecast=Y_train[best], periods=96, save=path_save_image + f"{args.model}_train_best.png")

        worst = np.argmax(losses_train)
        print("Worse index", worst)
        print(f"Worse rmse: {losses_train[worst]}")
        simple_plot(truth=y_train[worst,:fh], forecast=Y_train[worst], periods=96, save=path_save_image + f"{args.model}_train_worst.png")

        best = np.argmin(losses_valid)
        print(f"Best rmse: {losses_valid[best]}")
        simple_plot(truth=y_valid[best,:fh], forecast=Y_valid[best], periods=96, save=path_save_image + f"{args.model}_valid_best.png")

        worst = np.argmax(losses_valid)
        print(f"Worse rmse: {losses_valid[worst]}")
        simple_plot(truth=y_valid[worst,:fh], forecast=Y_valid[worst], periods=96, save=path_save_image + f"{args.model}_valid_worst.png")


