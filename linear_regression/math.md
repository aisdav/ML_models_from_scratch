# Linear Regression

## 1. Description of Linear Regression
Linear Regression is one of the fundamental algorithms in machine learning.  
It models the relationship between a target variable **y** and input features **X** using a linear function:  

![Linear model](https://latex.codecogs.com/svg.image?%5Chat%7By%7D%20%3D%20X%20%5Ccdot%20w%20%2B%20b)


where:  
- **w** — vector of weights (coefficients),  
- **X** — vector of features,  
- **b** — bias term.  

The goal of linear regression is to find the optimal weights and bias that **minimize a loss function**, typically:  
- **Mean Squared Error (MSE)**:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![MSE](https://latex.codecogs.com/svg.image?\color{white}\text{MSE}=\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2) 
- **Mean Absolute Error (MAE)**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![MAE](https://latex.codecogs.com/svg.image?\color{white}\text{MAE}=\frac{1}{m}\sum_{i=1}^{m}|y_i-\hat{y}_i|)


---

## 2. Methods to Train Linear Regression

### 2.1 Normal Equation
The weights can be computed **analytically** using the Normal Equation:  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![OLS solution](https://latex.codecogs.com/svg.image?\color{white}w%20%3D%20(X%5ET%20X)%5E%7B-1%7D%20X%5ET%20y)



**Pros:** exact solution, no iterations required.  
**Cons:** computationally expensive for large datasets because of matrix inversion.  

---

### 2.2 Gradient Descent
Gradient is vector that shows direction of the stepeest increase and how fast is this increase happening. Moving opposite to the gradient we decrease our loss function
Gradient Descent is an **iterative optimization method** that gradually minimizes the loss function by updating weights and bias in the opposite direction of the gradient:  

![Gradient Descent](https://latex.codecogs.com/svg.image?\color{white}w%5E%7B(t)%7D%20%3D%20w%5E%7B(t-1)%7D%20-%20%5Calpha%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20w%7D%2C%20%5Cquad%20b%5E%7B(t)%7D%20%3D%20b%5E%7B(t-1)%7D%20-%20%5Calpha%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20b%7D)


where:  
- \(J\) — loss function (e.g., MSE)  
- \(\alpha\) — learning rate  
- \(t\) — iteration step  

---

## 3. Gradient Descent Algorithm (Step-by-Step)

1. **Initialize** weights and bias (often zeros or small random numbers).  
2. **Compute predictions**: \(\hat{y} = X \cdot w + b\)  
3. **Compute gradients** of the loss with respect to weights and bias.  
4. **Update weights and bias**:  
![Gradient Descent Assignment](https://latex.codecogs.com/svg.image?\color{white}w%20:=%20w%20-%20%5Calpha%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20w%7D%2C%20%5Cquad%20b%20:=%20b%20-%20%5Calpha%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20b%7D)
 
5. **Repeat steps 2–4** until convergence (gradients are below a predefined threshold or maximum iterations reached).  
6. **Final prediction** is obtained using the learned weights and bias.  

---

## 4. Notes and Observations
- Choosing the right **learning rate** is crucial:  
  - Too large → weights oscillate or diverge.  
  - Too small → slow convergence.  
- Normal Equation is better for **small datasets**, Gradient Descent is better for **large datasets**.  
- For multiple features, both methods generalize to multidimensional X.  

