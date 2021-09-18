import numpy as np
import shap

class Explainer:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        _model_class = str(type(model))
        if "xgboost.sklearn.XGBClassifier" in _model_class:
            self.explainer = lambda b, f: shap.explainers.Tree(model, data=b, model_output="probability")(f)
        if "xgboost.sklearn.XGBRegressor" in _model_class:
            self.explainer = lambda b, f: shap.explainers.Tree(model, data=b, model_output="raw")(f)


    def shap_values(self, xb, xf=None):
        if xf is None:
            return np.absolute(self.explainer(xb, xb).values).mean(axis=0)
        else:
            return self.explainer(xb, xf).values


    def shap_values_ranking(self, xb, xf=None):
        return (-self.shap_values(xb, xf)).argsort().argsort()


    def shap_values_pop(self, xb, xf=None):
        xb_long = xb.reshape((xb.shape[0], xb.shape[1]*xb.shape[2]))
        if xf is None: # 2d
            return np.apply_along_axis(
                lambda x, d: self.shap_values(x.reshape((d[0], d[1]))),
                1, xb_long, d=xb[0].shape
            )
        else: # 1d
            return np.apply_along_axis(
                lambda x, f, d: self.shap_values(x.reshape((d[0], d[1])), f),
                1, xb_long, f=xf, d=xb[0].shape
            )


    def shap_values_ranking_pop(self, xb, xf=None):
        return (-self.shap_values_pop(xb, xf)).argsort(axis=1).argsort(axis=1)
