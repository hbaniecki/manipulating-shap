import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Algorithm:
    def __init__(
            self,
            explainer,
            constant=None,
            row_id=None,
            col_id=None
        ):

        self.explainer = explainer
        self._X = explainer.data.values
        self._n, self._p = self._X.shape

        if isinstance(row_id, int):
            self._x = explainer.data.values[row_id, :]
        else:
            if row_id is not None:
                warnings.warn("`row_id` is " + str(row_id) + " and should be an integer. Using `row_id=None`.")
            self._x = None
            
        if isinstance(col_id, int):
            self.col_id = [col_id]
        elif isinstance(col_id, list):
            self.col_id = col_id
        else:
            self.col_id = list(range(self._p))

        if constant is not None:
            self._idc = []
            for const in constant:
                self._idc.append(explainer.data.columns.get_loc(const))
        else:
            self._idc = None

        self.result_explanation = {'original': None, 'changed': None}
        self.result_data = None

        self.iter_losses = {'iter':[], 'loss':[], 'distance_importance':[], 'distance_ranking':[]}

    def fool(self, random_state=None):

        if random_state is not None:
            np.random.seed(random_state)

        self.result_explanation['original'] = self.explainer.shap_values(self._X, self._x)
        self.result_explanation['changed'] = np.zeros_like(self.result_explanation['original'])


    def fool_aim(self, target="auto", random_state=None):

        Algorithm.fool(self=self, random_state=random_state)
        
        if isinstance(target, np.ndarray):
            self.result_explanation['target'] = target
        else: # target="auto"
            self.result_explanation['target'] = np.repeat(
                self.result_explanation['original'].mean(),
                self.result_explanation['original'].shape[0]
            ) - self.result_explanation['original'] * 0.001


    #:# plots 
        
    def plot_data(self, i=0, constant=True, height=2, savefig=None):
        plt.rcParams["legend.handlelength"] = 0.1
        _colors = sns.color_palette("Set1").as_hex()[0:2][::-1]
        if i == 0:
            _df = self.result_data
        else:
            _data_changed = pd.DataFrame(self.get_best_data(i), columns=self.explainer.data.columns)
            _df = pd.concat((self.explainer.data, _data_changed))\
                    .reset_index(drop=True)\
                    .rename(index={'0': 'original', '1': 'changed'})\
                    .assign(dataset=pd.Series(['original', 'changed'])\
                                    .repeat(self._n).reset_index(drop=True))
        if not constant and self._idc is not None:
            _df = _df.drop(_df.columns[self._idc], axis=1)
        ax = sns.pairplot(_df, hue='dataset', height=height, palette=_colors)
        ax._legend.set_bbox_to_anchor((0.62, 0.64))
        if savefig:
            ax.savefig(savefig, bbox_inches='tight')
        plt.show()

    def plot_losses(self, lw=3, figsize=(9, 6), savefig=None):
        plt.rcParams["figure.figsize"] = figsize
        plt.plot(
            self.iter_losses['iter'], 
            self.iter_losses['loss'], 
            color='#000000', 
            lw=lw
        )
        plt.title('Learning curve', fontsize=20)
        plt.xlabel('epoch', fontsize=16)
        plt.ylabel('loss', fontsize=16)
        if savefig:
            plt.savefig(savefig)
        plt.show()

    def plot_explanation(self):
        import matplotlib.pyplot as plt
        temp = pd.DataFrame(self.result_explanation)
        x = np.arange(len(self.explainer.data.columns))
        width = 0.2
        fig, ax = plt.subplots()
        ax.bar(x - width, temp["original"], width, label='original', color="blue")
        ax.bar(x, temp['changed'], width, label='changed', color="red")
        ax.bar(x + width, temp['target'], width, label='target', color="black")
        ax.set_xticks(x)
        ax.legend()
        ax.set_xticklabels(self.explainer.data.columns)
        fig.tight_layout()
        plt.show()