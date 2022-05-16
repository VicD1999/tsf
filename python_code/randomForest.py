from util import rmse, get_dataset_sklearn, plot_results, load_sklearn_model, get_dataset_rnn, simple_plot
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import numpy as np
import pickle
import os

import argparse


if __name__ == '__main__':
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
                        type=str, default="RandomForestRegressor")

    args = parser.parse_args()

    models = {"RandomForestRegressor": RandomForestRegressor,
              "ExtraTreesRegressor": ExtraTreesRegressor}

    model = models[args.model]

    start = args.continue_training

    if args.training:
        for day in range(start, 96):
            X, y = get_dataset_sklearn(day=day, farm=0, type_data="train", gap=48, 
                                       history_size=96, forecast_horizon=96)
            print("X", X.shape)
            print("y", y.shape)

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
                with open(f'model/{args.model}/{args.model}_{day}.pkl','wb') as f:
                    pickle.dump(rfr, f)

            else:
                with open(f'model/{args.model}/{args.model}_{day}.pkl', 'rb') as f:
                    rfr = pickle.load(f)

            # y_hat = plot_results(model=rfr, 
            #                      X=X, y=y, 
            #                      save_path=f"results/RandomForest/{day}.png", 
            #                      sklearn=True, 
            #                      show=False)

        print("rmse:", rmse(y_hat, y))


    if args.evaluation:
        fh = 96
        day = 0

        X_train, y_train = get_dataset_sklearn(day=day, farm=0, type_data="train", gap=48, 
                                   history_size=96, forecast_horizon=96)
        X_valid, y_valid = get_dataset_sklearn(day=day, farm=0, type_data="valid", gap=48, 
                                   history_size=96, forecast_horizon=96)

        num_samples_train = X_train.shape[0]
        num_samples_valid = X_valid.shape[0]

        Y_train = np.empty((num_samples_train, fh))
        Y_valid = np.empty((num_samples_valid, fh))

        Y_train_truth = np.empty((num_samples_train, fh))
        Y_valid_truth = np.empty((num_samples_valid, fh))


        for day in range(fh):
            print("day", day)
            X_train, y_train = get_dataset_sklearn(day=day, farm=0, type_data="train", gap=48, 
                                       history_size=96, forecast_horizon=96, size=args.dataset_size)
            X_valid, y_valid = get_dataset_sklearn(day=day, farm=0, type_data="valid", gap=48, 
                                       history_size=96, forecast_horizon=96, size=args.dataset_size)
            
            print("X_train", X_train.shape)
            rfr = load_sklearn_model(path_to_model=f"model/{args.model}/{args.model}_{day}.pkl")        

            Y_train[:,day] = rfr.predict(X_train)
            Y_valid[:,day] = rfr.predict(X_valid)

            Y_train_truth[:,day] = y_train
            Y_valid_truth[:,day] = y_valid


        y_train = Y_train_truth
        y_valid = Y_valid_truth

        path_save_image = f"results/figure/{args.model}/"
        if not os.path.isdir(path_save_image):
            os.mkdir(path_save_image)

        losses_train = np.sqrt(np.mean(np.square(Y_train - y_train[:,:fh]), axis=1)) # [rmse(Y_train[i,:],y_train[i,:fh]) for i in range(num_samples_train)] # np.sqrt(np.mean(np.square(Y_train - y_train[:,:fh]), axis=1))
        simple_plot(truth=y_train[0,:fh], forecast=Y_train[0], periods=fh, save=path_save_image + f"{args.model}_train.png")
        print(f"rmse: {np.mean(losses_train):.4f} \pm {np.std(losses_train):.4f}")

        losses_valid = np.sqrt(np.mean(np.square(Y_valid - y_valid[:,:fh]), axis=1)) # [rmse(Y_valid[i,:],y_valid[i,:fh]) for i in range(num_samples_valid)]# np.sqrt(np.mean(np.square(Y_valid - y_valid[:,:fh]), axis=1))
        simple_plot(truth=y_valid[0,:fh], forecast=Y_valid[0], periods=fh, save=path_save_image + f"{args.model}_valid.png")
        print(f"rmse: {np.mean(losses_valid):.4f} \pm {np.std(losses_valid):.4f}")

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


