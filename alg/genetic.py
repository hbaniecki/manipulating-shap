import numpy as np
import pandas as pd
import tqdm

from . import algorithm     
from . import loss
from . import utils

class GeneticAlgorithm(algorithm.Algorithm):
    def __init__(
        self,
        explainer,
        constant=None,
        row_id=None,
        col_id=None,
        **kwargs
    ):
        super().__init__(
            explainer=explainer,
            constant=constant,
            row_id=row_id,
            col_id=col_id
        )
        
        params = dict(
            epsilon=1e-5,
            stop_iter=10,
            pop_count=50,
            std_ratio=1/9,
            mutation_prob=0.5,
            mutation_with_constraints=True,
            crossover_ratio=0.5,
            top_survivors=2,
            alpha=0.5
        )
        
        for k, v in kwargs.items():
            params[k] = v

        self.params = params
        
        # prepare std vector for mutation
        self._X_std = self._X.std(axis=0) * params['std_ratio']
        if self._idc is not None:
            for c in self._idc:
                self._X_std[c] = 0
        self._X_std_loss = np.where(self._X_std == 0, 1, self._X_std)
        
        if params['mutation_with_constraints']:
            self._X_minmax = {
                'min': np.amin(self._X, axis=0), 
                'max': np.amax(self._X, axis=0)
            }
        
        # calculate probs for rank selection method
        self._rank_probs = np.arange(params['pop_count'], 0, -1) /\
             (params['pop_count'] * (params['pop_count'] + 1) / 2)
        

    #:# algorithm
  
    def fool(
        self,
        max_iter=50,
        random_state=None,
        verbose=True,
        aim=False
    ):
        
        self._aim = aim
        if aim is False:
            super().fool(random_state=random_state)
            
        # init population
        self._X_pop = np.tile(self._X, (self.params['pop_count'], 1, 1))
        self._E_pop = np.tile(self.result_explanation['original'], (self._X_pop.shape[0], 1)) 
        self._L_pop = np.zeros(self.params['pop_count']) 
        self.mutation(adjust=3)
        self.append_losses(i=0)
        
        pbar = tqdm.tqdm(range(1, max_iter + 1), disable=not verbose)
        for i in pbar:
            self.crossover()
            self.mutation()
            self.evaluation()
            if i != max_iter:
                self.selection()

            self.append_losses(i=i)
            pbar.set_description("Iter: %s || Loss: %s" % (i, self.iter_losses['loss'][-1]))
            if utils.check_early_stopping(self.iter_losses, self.params['epsilon'], self.params['stop_iter']):
                break

        self.result_explanation['changed'] = self.explainer.shap_values(self.get_best_data(), self._x)

        _data_changed = pd.DataFrame(self.get_best_data(), columns=self.explainer.data.columns)
        
        self.result_data = pd.concat((self.explainer.data, _data_changed))\
            .reset_index(drop=True)\
            .rename(index={'0': 'original', '1': 'changed'})\
            .assign(dataset=pd.Series(['original', 'changed'])\
                            .repeat(self._n).reset_index(drop=True))

    def fool_aim(
            self,
            target="auto",
            max_iter=50,
            random_state=None,
            verbose=True
        ):
        super().fool_aim(
            target=target,
            random_state=random_state
        )
        self.fool(
            max_iter=max_iter, 
            random_state=random_state, 
            verbose=verbose, 
            aim=True
        )
            

    #:# inside
    
    def mutation(self, adjust=1):   
        _temp_pop_count = self._X_pop.shape[0]         
        _theta = np.random.normal(
            loc=0,
            scale=self._X_std * adjust,
            size=(_temp_pop_count, self._n, self._p)
        )
        # preserve zeros
        _theta = np.where(self._X_pop == 0, 0, _theta)
        # column mask made with the probability 
        _mask = np.random.binomial(
            n=1,
            p=self.params['mutation_prob'], 
            size=(_temp_pop_count, 1, self._p)
        )
        self._X_pop += _theta * _mask
        
        if self.params['mutation_with_constraints']:
            # add min/max constraints for the variable distribution
            # this feature may lead to a much longer computation time
            _X_pop_long = self._X_pop.reshape(_temp_pop_count * self._n, self._p)
            _X_long = np.tile(self._X, (_temp_pop_count, 1))
            for i in range(self._p):
                _max_mask = _X_pop_long[:, i] > self._X_minmax['max'][i]
                _min_mask = _X_pop_long[:, i] < self._X_minmax['min'][i]
                _X_pop_long[:, i][_max_mask] = np.random.uniform(
                    _X_long[:, i][_max_mask],
                    np.repeat(self._X_minmax['max'][i], _max_mask.sum())
                )
                _X_pop_long[:, i][_min_mask] = np.random.uniform(
                    np.repeat(self._X_minmax['min'][i], _min_mask.sum()),
                    _X_long[:, i][_min_mask]
                )
            self._X_pop = _X_pop_long.reshape(_temp_pop_count, self._n, self._p)
    
    def crossover(self):
        # indexes of subset of columns (length is between 0 and p/2)
        _idv = np.random.choice(
            np.arange(self._p),
            size=np.random.choice(int(self._p / 2)),
            replace=False
        )
        # indexes of subset of population
        _idpop = np.random.choice(
            self.params['pop_count'], 
            size=int(self.params['pop_count'] * self.params['crossover_ratio']),
            replace=False
        )
        # get shuffled population
        _childs = self._X_pop[_idpop, :, :]
        # swap columns
        _childs[:, :, _idv] = _childs[::-1, :, _idv]
            
        self._X_pop = np.concatenate((self._X_pop, _childs))
    
    def evaluation(self):
        self._E_pop = self.explainer.shap_values_pop(self._X_pop, self._x)
        _loss = loss.loss_pop(
            arr1=self.result_explanation['target'] if self._aim else self.result_explanation['original'],
            arr2=self._E_pop,
            col_id=self.col_id
        )
        a = self.params['alpha']
        self._L_pop = a*_loss[0] + (1-a)*_loss[1] if self._aim else -a*_loss[0] - (1-a)*_loss[1]
            
    def selection(self):
        #:# take n best individuals and use p = i/(n*(n-1))
        _top_survivors = self.params['top_survivors']
        _top_f_ids = np.argpartition(self._L_pop, _top_survivors)[:_top_survivors]
        _random_ids = np.random.choice(
            self.params['pop_count'], 
            size=self.params['pop_count'] - _top_survivors, 
            replace=True,
            p=self._rank_probs
        )
        _sorted_ids = np.argsort(self._L_pop)[_random_ids]
        self._X_pop = np.concatenate((
            self._X_pop[_sorted_ids],
            self._X_pop[_top_f_ids]
        ))
        self._L_pop = np.concatenate((
            self._L_pop[_sorted_ids],
            self._L_pop[_top_f_ids]
        ))
        assert self._X_pop.shape[0] == self.params['pop_count'], 'wrong selection'
    

    #:# helper
    
    def get_best_id(self, i=0):
        return np.argsort(self._L_pop)[i]
        
    def get_best_data(self, i=0):
        return self._X_pop[np.argsort(self._L_pop)][i]

    def append_losses(self, i=0):
        _best = self.get_best_id()
        _loss = loss.loss(
            arr1=self.result_explanation['target'] if self._aim else self.result_explanation['original'],
            arr2=self._E_pop[_best],
            col_id=self.col_id
        )
        a = self.params['alpha']
        self.iter_losses['iter'].append(i)
        self.iter_losses['loss'].append(a*_loss[0] + (1-a)*_loss[1] if self._aim else -a*_loss[0] - (1-a)*_loss[1])
        self.iter_losses['distance_importance'].append(_loss[0])
        self.iter_losses['distance_ranking'].append(_loss[1])
