import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import inspect

class PCAGuidedFeatureBuilderClass():
    def __init__(self, X: pd.DataFrame, method: str="mean", model=None,\
                 y: pd.Series=None, test_size=0.2, scoring_method=None,\
                    random_state: int=None, verbose: bool=False):
        #-----------------------------------------------------------------------
        # Verification of required arguments:
        
        # Verify X type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        self.X = X.copy()

        # Verify method type
        if not isinstance(method, str):
            raise TypeError("method must be a string of the method to be used (e.g., 'mean')")
        # Only one method is available at the moment:
        if method != "mean":
            raise ValueError("'mean' is the only method implemented right now.")
        self.method = method
        #-----------------------------------------------------------------------

        #-----------------------------------------------------------------------
        # Verification of optional arguments:

        # Verify model if one has been specified
        if model is not None:
            if not hasattr(model,"fit") or not callable(getattr(model,"fit")):
                raise TypeError("Model must implement a callable 'fit(X,y)' method.")
            if not hasattr(model,"predict") or not callable(getattr(model,"predict")):
                raise TypeError("Model must implement a callable 'predict(X)' method.")
        self.model = model
        
        # Verifiy y if y has been specified 
        if y is not None:
            if not isinstance(y, pd.Series):
                raise TypeError("If specified, y must be a pandas Series.")
        self.y = y

        # Verify test_size in domain. The default value will succeed this check.
        if (test_size <= 0) or (1 <= test_size):
            raise ValueError("test_size must be a value between 0 and 1, non-inclusive.")
        self.test_size = test_size

        # Verify scoring_method if one has been specified.
        if scoring_method is not None:
            if not callable(scoring_method) and not (scoring_method==None):
                raise TypeError("If specified, the scoring method must be a callable function used to score the model.")
            sig = inspect.signature(scoring_method)
            params = sig.parameters
            if len(params) < 2:
                raise ValueError("scoring_method must accept two arguments (y_true, y_pred).")
        self.scoring_method = scoring_method
        if random_state is not None:
            if not isinstance(random_state, int):
                raise TypeError("random_state must be an integer.")
        self.random_state = random_state
        #-----------------------------------------------------------------------
        self.will_be_scoring = False
        if model is not None and y is not None and scoring_method is not None\
            and random_state is not None:
            self.will_be_scoring = True
        #-----------------------------------------------------------------------
        # Internals
        self.verbose                    = verbose
        self.PCA_performed              = False
        self.dropped_low_PCAs           = False
        self.ran_engineer_features    = False


    @staticmethod
    def help():
        print("\nGuidedPCAFeatureBuilder is intended to assist the end-user in")
        print("determining relevant features for a dataset and automatically")
        print("performing analysis on simple combinations of those features.\n")
        print("GuidedPCAFeatureBuilder will experience poor performance if")
        print("large, raw datasets are simply passed to it. End-users are")
        print("recommended to perform other feature selection methods to")
        print("identify high-quality candidate features.")
        print("Furthermore, GuidedPCAFeatureBuilder does no preprocessing,")
        print("cannot handle missing values, and cannot handle categorical")
        print("inputs. End-users must account for this and perform scaling,")
        print("encoding, missing value handling, etc. before using this tool.")
        print("---------------------------------------------------------------")
        print("GuidedPCAFeatureBuilder requires the following information:")
        print("X                - the dataset as a pandas DataFrame.")
        print("method           - the method used to cut off features that do")
        print("                     not meet the threshold of the method. For")
        print("                     example, 'mean' cuts off all absolute PCA")
        print("                     values that are below the mean of the")
        print("                     absolute values of the values in that PCA")
        print("                     column. method should be a string.")
        print("---------------------------------------------------------------")
        print("GuidedPCAFeatureBuilder has optional properties:")
        print("model            - the model used to score the original dataset")
        print("                     and the features determined by the feature")
        print("                     builder. The user has the option to take")
        print("                     the features and score their model")
        print("                     themself. The model must be a callable")
        print("                     object with .fit(X,y) and .predict(X)")
        print("                     methods.")
        print("y                - the target values as a pandas Series. This")
        print("                     is necessary for the feature builder to")
        print("                     calculate the score itself.")
        print("test_size        - the test_size ratio. The proportion of X and")
        print("                     y that will be witheld as test data.")
        print("                     0 < test_size < 1")
        print("scoring_method   - the method used to score the results of the")
        print("                     model. For example, MSE(y_test, y_pred).")
        print("                     scoring_method must take two arguments.")
        print("---------------------------------------------------------------")
        print("After instantiating PCAGuidedBuilder")
        print("(e.g., builder=PCAGuidedBuilder(X)), call the 'main' function.")
        print("PCAGuidedFeatureBuilder.main() will return the engineered")
        print("features. If it has enough of the arguments to perform a")
        print("scoring analysis, it will score before returning the engineered")
        print("features, the MAE for the base model, and the MAE with all")
        print("automatically-generated features.")
        print("PCAGuidedFeatureBuilder supports a .Transform(X) method, where")
        print("once the original features have been developed by the method,")
        print("subsequent features can be generated with the same indices.")
        print("The .Transform(X) method returns only the new features.")


    def PerformPCA(self):
        # Because this method can only be called once the Feature Builder is
        # initialized and it will only initialized if X is pandas DataFrame,
        # no further verification is required.
        pca = PCA()
        X_pca = pca.fit_transform(self.X)
        component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        X_pca = pd.DataFrame(X_pca, columns=component_names)
        self.loadings = pd.DataFrame(pca.components_.T,\
                                columns=component_names,\
                                index=self.X.columns)
        self.PCA_performed = True

    def DropLowPCAs(self):
        if not self.PCA_performed:
            raise RuntimeError("Cannot drop low PCA values before PCA is performed. Most end-users should not be seeing this.")
        
        def get_high_loading_features(loadings: pd.DataFrame) -> dict[str, pd.Series]:
            """
            For each principal component in 'loadings', return a Series of
            features whose absolute loadings are greater than or equal to the
            mean absolute loading."""
            if self.method == 'mean':
                return {
                    col: loadings[col]
                    .sort_values(ascending=False)
                    .loc[loadings[col].abs() >= loadings[col].abs().mean()]
                    for col in loadings.columns
                }
            else:
                raise ValueError("method not recognized. 'mean' is the only functioning method for now.")
        
        self.high_loading_features = get_high_loading_features(self.loadings)
        self.dropped_low_PCAs = True
    
    def EngineerFeatures(self):
        if not self.PCA_performed:
            raise RuntimeError("Cannot engineer features until PCA is performed. Most end-users should not be seeing this.")
        if not self.dropped_low_PCAs:
            raise RuntimeError("Cannot engineer features until low PCAs are dropped. Most end-users should not be seeing this.")
        
        EngineeredFeatures = pd.DataFrame(index=self.X.index)
        self.signed_coeffs = {}

        # Collect sign-based coefficients
        for comp_name, pca_series in self.high_loading_features.items():
            if pca_series.index.nunique() == 1:
                if self.verbose:
                    print(f"* {comp_name} has only one high-loading feature ({pca_series.index[0]}).")
                    print("This feature will already be in the model. No engineered feature created.")
            else:
                signs = pca_series / pca_series.abs()
                self.signed_coeffs[comp_name] = signs

        # Create engineered features
        for comp_name, coeffs in self.signed_coeffs.items():
            feature_name = f"Engineered_{comp_name}"
            EngineeredFeatures[feature_name] = sum(
                coeff*self.X[feat] for feat, coeff in coeffs.items()
            )
        self.ran_engineer_features = True
        self.EngineeredFeatures = EngineeredFeatures

    def Transform(self, X_to_transform):
        if not self.PCA_performed:
            raise RuntimeError("Cannot transform until PCA is performed. Run PCAGuidedFeatureBuilder.main()")
        if not self.dropped_low_PCAs:
            raise RuntimeError("Cannot transform until low PCAs are dropped. Run PCAGuidedFeatureBuilder.main()")
        if not self.ran_engineer_features:
            raise RuntimeError("Cannot transform until features are engineered. Run PCAGuidedFeatureBuilder.main()")
        
        TransformedFeatures = pd.DataFrame(index=X_to_transform.index)
        for comp_name, coeffs in self.signed_coeffs.items():
            feature_name = f"Engineered_{comp_name}"
            #missing_feats = [feat for feat in coeffs if feat not in X_to_transform.columns]
            #if missing_feats:
                #raise KeyError(f"Missing expected features in transform: {missing_feats}")
            TransformedFeatures[feature_name] = sum(coeff*X_to_transform[feat] for feat, coeff in coeffs.items())
        return TransformedFeatures
    
    def PerformScoringAnalysis(self):
        if not self.PCA_performed:
            raise RuntimeError("Cannot perform scoring analysis until PCA is performed. Most end-users should not be seeing this.")
        if not self.dropped_low_PCAs:
            raise RuntimeError("Cannot perform scoring analysis until low PCAs are dropped. Most end-users should not be seeing this.")
        if not self.ran_engineer_features:
            raise RuntimeError("Cannot perform scoring analysis until features are engineered. Most end-users should not be seeing this.")

        def GetMAEwithDataSet(internal_X: pd.DataFrame):
            X_train, X_test, y_train, y_test = train_test_split(internal_X, self.y, test_size= self.test_size, random_state=self.random_state)
            self.model.fit(X_train,y_train)
            preds = self.model.predict(X_test)
            return mean_absolute_error(y_test, preds)

        #Base model:
        self.MAE_base_model = GetMAEwithDataSet(self.X)
        if self.verbose:
            print("Baseline MAE for the provided model: ", self.MAE_base_model)

        #With all engineered features:
        X_all_features_scoring_analysis = pd.concat([self.X.copy(),self.EngineeredFeatures],axis=1)
        self.MAE_all_features = GetMAEwithDataSet(X_all_features_scoring_analysis)
        improvement_percent = 100*(self.MAE_base_model-self.MAE_all_features)/self.MAE_base_model
        if self.verbose:
            print("MAE with all engineered features: ", self.MAE_all_features)
            print(f"Relative MAE improvement (%): {improvement_percent:2f}")
    
    def main(self):
        self.PerformPCA()
        self.DropLowPCAs()
        self.EngineerFeatures()

        if self.will_be_scoring:
            self.PerformScoringAnalysis()
            return self.EngineeredFeatures, self.MAE_base_model, self.MAE_all_features
        else:
            return self.EngineeredFeatures