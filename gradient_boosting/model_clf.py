class GBMClassifier:
  def __init__(self,logitboost = false, n_estimators =100,max_depth=3, random_state = 0):
    self.logitboost = logiboost
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.random_state = random_state
    
  def _softmax(self, predictions):
    exp = np.exp(predictions)
    return exp/np.sum(exp,axis,keepdims = True)
  def _compute_gammas(self,residuals,leaf_indexes,eps =1e-4):
    gammas = []
    for j in np.unique(leaf_indexes):
      x_i = np.where(leaf_indexes)
      numerator = np.sum(residuals[x_i])
      norm_residuals_xi = np.linalg.norm(residuals[x_i])+eps
      denominator =np.sum(norm_residuals_xi * (1-norm_residuals_xi))
      gamms = (self.K-1)/self.K* numerator / denominator
      gammas.append(gamma)
    return gammas
  def fit(self,X,y):
    self.K =len(np.unique(y))
    self.trees= {self.K [] for k in range(self.K)}
    one_hot_y =pd.get_dummies(y).to_numpy()
    predictions = np.zeros(one_hot_y.shape)
    for _ in range(self.n_estimators):
      probabilitis =self._softmax(predictions)
      for k in range(self.K):
        if self.logitboost:
          numerator = (one_hot_y.T[k]-probabilities.T[k])
          denominator= probabilities.T[k]*(probabilities.T[k])
          residuals = (self.K-1)/self.K *numerator/denominator
          weights = denominator
        else:
          residual = one_hot_y.T[k] - probabilities.T[k]
          weights = None
        tree = DecisionTreeRegressor(criterion = 'friedman_mse',max_depth =self.max_depth,self.random_state = random_state)
        tree.fit(X,residuals,sample_weights=weights)
        self.trees[k].append(tree)
        leaf_indexes= tree.apply(X)
        gammas = [] if self.logitboost else self._compute_gammas(residual,leaf_indexes)
        predictiions.T[k]+=self.learning_rate * tree.predict(X)+ np.sum(gammas)
    
    def predict(self,samples):
      predictions = np.zeros((len(samples),self.K))
      for i in range(self.n_estimators):
        for k in range(self.K):
          predictions.T[k]+=self.learning_rate * self.trees[k][i].predict(samples)
      return np.argmax(predictions,axis=1)
