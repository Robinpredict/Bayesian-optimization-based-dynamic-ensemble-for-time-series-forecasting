import itertools
from hyperopt import hp, tpe, partial, fmin
from sklearn.metrics import mean_absolute_error as MAE
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class ensemble_Forecast(object):  #ITVP,ycand,y
    def __init__(self, OOB_valid_pair, inpool_forecast, ground_truth):
        self.pool_size, self.forecast_length = inpool_forecast.shape
        self.inpool_forecast = inpool_forecast
        self.OOB_inpool_Valid, self.OOB_Valid = OOB_valid_pair
        self.ground_truth = ground_truth


    def dynamic_ensemble_forecast(self, specified=False, specification=[],without_OOB=False,BOA=0,rcount=100):
        if specified == True:
            cool_start = specification[0]
            evaluation_window = specification[1]
            weighting_pool = specification[2]
            metric = specification[3]
            weighting_method = specification[4]  # 'inverted','squared inverted','Softmax'
            discount = specification[5]
            alpha = specification[6]
        else:
            cool_start, evaluation_window, weighting_pool, metric, weighting_method,discount,alpha = self.dynamic_ensemble_tuning(without_OOB=without_OOB, BOA=BOA, rcount=rcount)

        HP = [cool_start, evaluation_window, weighting_pool, metric, weighting_method,discount,alpha]
        ensemble_hat, Ensemble_Matrix = ensemble_Forecast.dynamic_ensemble_forecasting_method(self.inpool_forecast,
                                                                                              self.ground_truth,
                                                                                              *HP)
        return ensemble_hat, Ensemble_Matrix, HP

    def dynamic_ensemble_tuning(self, without_OOB, BOA, rcount):
        if BOA != 0:
            best_params = self.BOA_tuning(BOA, without_OOB)
            best_HP = [
                int(best_params['cool_start']),
                int(best_params['evaluation_window']),
                int(best_params['weighting_pool']),
                ['MAE', 'MSE'][best_params['metric']],
                ['Inverted', 'SquaredInverted', 'Softmax'][best_params['weighting_method']],
                best_params['discount'],
                best_params['alpha']
            ]
        else:
            cool_start = np.arange(self.pool_size)
            evaluation_window = np.arange(1,13)
            weighting_pool = np.arange(1, self.pool_size+1)
            metric = ['MAE','MSE']
            weighting_method = ['Inverted', 'SquaredInverted',
                                'Softmax']
            discount = [1.05, 1.1, 1.2, 1.3]
            alpha = [0, 0.25, 0.5, 0.75, 1.0]
            HParams = list(itertools.product(cool_start, evaluation_window,  weighting_pool, metric,weighting_method, discount, alpha))
            best_vloss = 1e6
            best_HP = HParams[0]
            M = self.OOB_Valid.shape[0]
            for count in range(rcount):
            #for HP in HParams:
                HP = HParams[np.random.randint(0,len(HParams))]
                loss_HP = 0
                if without_OOB is False:
                    for m in range(M):
                        valid_ground_truth_hat, _ = ensemble_Forecast.dynamic_ensemble_forecasting_method(
                            self.OOB_inpool_Valid[m, :, :], self.OOB_Valid[m, :], *HP)
                        loss_HP += MAE(valid_ground_truth_hat, self.OOB_Valid[m, :])
                else:
                    valid_ground_truth_hat, _ = ensemble_Forecast.dynamic_ensemble_forecasting_method(
                        self.OOB_inpool_Valid[-1, :, :],
                        self.OOB_Valid[-1, :],
                        *HP)
                    loss_HP = MAE(valid_ground_truth_hat, self.OOB_Valid[4, :])

                if loss_HP < best_vloss:
                    best_vloss = loss_HP
                    best_HP = HP
                if count%100==0:
                    print(count,'Without_OOB =', without_OOB, ', cool_start=', best_HP[0], ', evaluation_window=', best_HP[1], ', weighting_pool=',
                          best_HP[2], ', metric=', best_HP[3], ', weighting_method=', best_HP[4], 'discount=',best_HP[5], 'alpha=',best_HP[6])

        return best_HP

    def BOA_tuning(self, BOA, without_OOB):
        space = {
            'cool_start': hp.quniform('cool_start', 0, self.pool_size-1, 1),
            'evaluation_window': hp.quniform('evaluation_window', 1, 13, 1),
            'weighting_pool': hp.quniform('weighting_pool', 1, self.pool_size, 1),
            'metric': hp.choice('metric', ['MAE', 'MSE']),
            'weighting_method': hp.choice('weighting_method', ['Inverted', 'SquaredInverted', 'Softmax']),
            'discount':hp.uniform('discount',1.0,1.5),
            'alpha': hp.uniform('alpha', 0.0, 1),

        }
        algo = partial(tpe.suggest, n_startup_jobs=1)
        if without_OOB == True:
            best = fmin(self.BOA_eval, space, algo=algo, max_evals=BOA, verbose=True
                    , rstate=np.random.RandomState(1))
        else:
            best = fmin(self.BOA_eval_OOB, space, algo=algo, max_evals=BOA, verbose=True
                        , rstate=np.random.RandomState(1))
        return best

    def BOA_eval(self, argsDict):
        HP = {
            'cool_start': int(argsDict['cool_start']),
            'evaluation_window': int(argsDict['evaluation_window']),
            'weighting_pool': int(argsDict['weighting_pool']),
            'metric': argsDict['metric'],
            'weighting_method': argsDict['weighting_method'],
            'discount': argsDict['discount'],
            'alpha': argsDict['alpha'],
        }
        valid_ground_truth_hat, _ = ensemble_Forecast.dynamic_ensemble_forecasting_method(
            self.OOB_inpool_Valid[-1, :, :],
            self.OOB_Valid[-1, :],
            **HP)
        loss_HP = MAE(valid_ground_truth_hat, self.OOB_Valid[-1, :])
        return loss_HP

    def BOA_eval_OOB(self, argsDict):
        HP = {
            'cool_start': int(argsDict['cool_start']),
            'evaluation_window': int(argsDict['evaluation_window']),
            'weighting_pool': int(argsDict['weighting_pool']),
            'metric': argsDict['metric'],
            'weighting_method': argsDict['weighting_method'],
            'discount': argsDict['discount'],
            'alpha': argsDict['alpha'],
        }
        loss_HP = 0
        M = self.OOB_Valid.shape[0]

        for m in range(M):
            valid_ground_truth_hat, _ = ensemble_Forecast.dynamic_ensemble_forecasting_method(
                self.OOB_inpool_Valid[m, :, :], self.OOB_Valid[m, :], **HP)
            loss_HP += MAE(valid_ground_truth_hat, self.OOB_Valid[m, :])

        return loss_HP

    @staticmethod
    def dynamic_weight_calculation(window_difference, cand_pool, weighting_pool, metric, weighting_method, discount_factor):
        #window_difference is inpool_forecast x evaluation_window
        assert metric in ['MSE', 'MAE']
        if metric is 'MSE':
            try:
                _,tau = window_difference.shape
                discount = np.power(discount_factor, np.arange(1,tau+1))
                discounted_window_difference = np.multiply(window_difference, discount)
                column_error = np.mean(np.power(discounted_window_difference,2), axis=1)
            except:
                column_error = np.power(window_difference,2)
        if metric is 'MAE':
            try:
                _,tau = window_difference.shape
                discount = np.power(discount_factor, np.arange(1,tau+1))
                discounted_window_difference = np.multiply(window_difference, discount)
                column_error = np.mean(discounted_window_difference, axis=1)
            except:
                column_error = window_difference

        column_error = np.array([1e-5 if i<1e-12 else i for i in column_error])


        assert weighting_method in ['Softmax', 'Inverted', 'SquaredInverted']
        if weighting_method is 'Softmax':
            if np.sum(np.exp(-column_error)) <1e-10:
                column_error = column_error/np.min(column_error)
            column_weight_all = np.exp(-column_error) / np.sum(np.exp(-column_error))

        if weighting_method is 'Inverted':
            column_weight_all = np.power(column_error,-1) / np.sum(np.power(np.abs(column_error),-1))

        if weighting_method is 'SquaredInverted':

            column_weight_all = np.power(column_error,-2) / np.sum(np.power(column_error,-2))

        column_weight_all = np.nan_to_num(column_weight_all)
        idx = np.argpartition(column_weight_all, -weighting_pool)[-weighting_pool:]
        column_weight = np.array([column_weight_all[i] if i in idx else 0 for i in range(cand_pool)])
        column_weight = column_weight / np.sum(column_weight)

        assert np.linalg.norm(np.sum(column_weight) - 1.0) < 1e-3
        return column_weight

    @staticmethod
    def dynamic_ensemble_matrix_computation(inpool_forecast, ground_truth, cool_start, evaluation_window, weighting_pool, metric, weighting_method,discount,alpha):
        Difference_Matrix = np.abs(inpool_forecast - ground_truth)  #每一行减
        pool_size, forecast_length = inpool_forecast.shape
        Ensemble_Matrix = np.zeros((pool_size, forecast_length))
        for i in range(forecast_length):
            if i <= evaluation_window-1:
                if i is 0:
                    columns_weight = np.zeros(pool_size)   # inpool models
                    columns_weight[cool_start] = 1
                    Ensemble_Matrix[:, i] = columns_weight
                    continue
                else:

                    window_difference = Difference_Matrix[:,:i]
            if i > evaluation_window-1:
                window_difference = Difference_Matrix[:, i-evaluation_window:i]
            columns_weight = ensemble_Forecast.dynamic_weight_calculation(window_difference,pool_size, weighting_pool, metric, weighting_method,discount)
            Ensemble_Matrix[:,i] = columns_weight

        Smooth_Matrix = np.zeros_like(Ensemble_Matrix)
        for i in range(0, Ensemble_Matrix.shape[1]):
            if i == 0:
                Smooth_Matrix[:, i] =Ensemble_Matrix[:, i]
            else:
                Smooth_Matrix[:, i] = alpha * Ensemble_Matrix[:, i - 1] + (1 - alpha) * Ensemble_Matrix[:, i]

        return Smooth_Matrix

    @staticmethod
    def dynamic_ensemble_forecasting_method(inpool_forecast, ground_truth,cool_start, evaluation_window, weighting_pool, metric, weighting_method,discount,alpha):

        Ensemble_Matrix =  ensemble_Forecast.dynamic_ensemble_matrix_computation(inpool_forecast, ground_truth, cool_start, evaluation_window, weighting_pool, metric,
                                                         weighting_method,discount,alpha)
        Ensemble_forecast = []
        pool_size, forecast_length = inpool_forecast.shape
        for i in range(forecast_length):
            forecast_i = np.dot(Ensemble_Matrix[:, i], inpool_forecast[:, i])
            Ensemble_forecast.append(forecast_i)
        return np.array(Ensemble_forecast), Ensemble_Matrix

def main():
    print('Program starts..')
    '''
    As the complete project requires the fine tuning of the model candidates, we can change the model candidtaes 
    of our interests. Therefore, we only share the part of the dynamic ensemble framework. 
    '''
    Test_len = 100
    Val_len = 100
    Train_len = 800
    T = Test_len + Val_len + Train_len

    Model_candidates = 12  # ten model candidates
    OOB_pair_nums = 8  # the number of shifts on training data to create multiple OOB pairs, we by default take it as 5
    slide_ratio = 0.2
    l_slide = int(slide_ratio * Val_len)

    # TS data
    TS = np.expand_dims(np.sin(T * np.arange(T)), axis=0)
    y_test = TS[:, -Test_len:]  # test data, shape:1xTest_len
    y_can_test = y_test + 0.1 * np.random.randn(Model_candidates,
                                                Test_len)  # prediction results of each independent model candidate,shape:mxT

    OOB_valid = []
    OOB_can_valid = []
    for i in range(OOB_pair_nums):
        OOB_valid.append(TS[:, Train_len - l_slide * i:Train_len + Val_len - l_slide * i])
        OOB_can_valid.append(np.expand_dims(
            TS[:, Train_len - l_slide * i:Train_len + Val_len - l_slide * i] + 0.1 * np.random.randn(Model_candidates,
                                                                                                     Val_len), axis=0))

    OOB_valid = np.vstack(OOB_valid)  # OOB_valid store the validation data with some shifts for augmentation
    OOB_can_valid = np.vstack(
        OOB_can_valid)  # OOB_can_valid sotres the predictions of each model candidate for each augmented version of validation data
    OOB_valid_pair = [OOB_can_valid, OOB_valid]
    # no tuning, no augmentation on validation, with specified_configuration

    specification = [0, 12, Model_candidates, 'MSE', 'Inverted', 1, 0]
    model = ensemble_Forecast(OOB_valid_pair=OOB_valid_pair, inpool_forecast=y_can_test,
                              ground_truth=y_test)
    ensemble_pre, Ensemble_Matrix, HP = model.dynamic_ensemble_forecast(specified=True, specification=specification)

    # with BOA tuning by setting BOA, no augmentation
    model2 = ensemble_Forecast(OOB_valid_pair=OOB_valid_pair, inpool_forecast=y_can_test,
                               ground_truth=y_test)
    ensemble_pre, Ensemble_Matrix, HP = model2.dynamic_ensemble_forecast(specified=False, without_OOB=True, BOA=100)

    # with random search by setting rcount, with augmentation on validation data
    model3 = ensemble_Forecast(OOB_valid_pair=OOB_valid_pair, inpool_forecast=y_can_test,
                               ground_truth=y_test)
    ensemble_pre, Ensemble_Matrix, HP = model3.dynamic_ensemble_forecast(specified=False, without_OOB=False,
                                                                         rcount=1000)


if __name__ == '__main__':
    main()
