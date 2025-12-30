# Linear Regression

## 1. Description of Linear Regression
Linear Regression is one of the fundamental algorithms in machine learning.  
It models the relationship between a target variable **y** and input features **X** using a linear function:  

\[
\hat{y} = X \cdot w + b
\]

where:  
- **w** — vector of weights (coefficients),  
- **X** — vector of features,  
- **b** — bias term.  

The goal of linear regression is to find the optimal weights and bias that **minimize a loss function**, typically:  
- **Mean Squared Error (MSE)**:  
\[
\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
\]  
- **Mean Absolute Error (MAE)**

![MAE](https://latex.codecogs.com/svg.image?\color{white}\text{MAE}=\frac{1}{m}\sum_{i=1}^{m}|y_i-\hat{y}_i|)


---

## 2. Methods to Train Linear Regression

### 2.1 Normal Equation
The weights can be computed **analytically** using the Normal Equation:  

\[
w = (X^T X)^{-1} X^T y
\]

**Pros:** exact solution, no iterations required.  
**Cons:** computationally expensive for large datasets because of matrix inversion.  

---

### 2.2 Gradient Descent
Gradient is vector that shows direction of the stepeest increase and how fast is this increase happening. Moving opposite to the gradient we decrease our loss function
Gradient Descent is an **iterative optimization method** that gradually minimizes the loss function by updating weights and bias in the opposite direction of the gradient:  

\[
w^{(t)} = w^{(t-1)} - \alpha \frac{\partial J}{\partial w}, \quad
b^{(t)} = b^{(t-1)} - \alpha \frac{\partial J}{\partial b}
\]

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
\[
w := w - \alpha \frac{\partial J}{\partial w}, \quad
b := b - \alpha \frac{\partial J}{\partial b}
\]  
5. **Repeat steps 2–4** until convergence (gradients are below a predefined threshold or maximum iterations reached).  
6. **Final prediction** is obtained using the learned weights and bias.  

---

## 4. Notes and Observations
- Choosing the right **learning rate** is crucial:  
  - Too large → weights oscillate or diverge.  
  - Too small → slow convergence.  
- Normal Equation is better for **small datasets**, Gradient Descent is better for **large datasets**.  
- For multiple features, both methods generalize to multidimensional X.  

