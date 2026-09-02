# **ECE2112_PA2**
#### Jeremy Rafael G. Lacuna | 2ECE-C
*This repository contains three programming problems which covers **Module 2 - Numpy**.* <br>
<br>
**Objectives:**
1. Create and reshape NumPy arrays using appropriate NumPy functions;
2. Perform vectorized numerical operations on an ndarray;
3. Compute array statistics and use Boolean conditions to select elements; and
4. Save computed NumPy arrays as .npy files.
## A. **Reproducible Normalization Problem**
Create a reproducible random 5 × 5 integer **ndarray** named **X**. Use the following two statements before performing any calculation:
`np.random.seed(2112)` <br>
`X = np.random.randint(10, 101, size=(5, 5))` <br>
Normalize the complete array using:
$$Z = \frac{X - \bar{x}}{\sigma}$$ 
<br> where $\bar{x}$ is the mean of all 25 elements and $\sigma$ is their population standard deviation as returned by NumPy's default **std()** call. Store the normalized array in **X_normalized**.
Functions used for this problem:
- **np.random.seed()**: Used to initialize the random number generator. **Makes random results predictable and reproducible**.
- **np.random.randint()**: Used to **generate random integers** within a specified range.
- **np.mean() & np.std()**: Computes the **mean** and the **standard deviation** of the array elements.
### **Code:**
```python
import numpy as np

def reproducible_normalization():
    np.random.seed(2112)
    X = np.random.randint (10, 101, size=(5, 5))

    X_normalized = (X - np.mean(X)) / np.std(X)
    np.save('X_normalized.npy', X_normalized)

    return X, X_normalized, np.mean(X_normalized), np.std(X_normalized)
```
#### **Test Cases:**
```python
X, X_norm, mean, std = reproducible_normalization()
```
```python
print("X:\n", X)
```
Output: <br>
```
X:
 [[48 11 15 67 21]
 [11 41 13 66 24]
 [71 79 53 67 70]
 [77 35 91 19 96]
 [35 54 37 41 17]]
```
```python
print("\nX_normalized:\n", X_norm)
```
Output: <br>
```
X_normalized:
 [[ 0.06340841 -1.36714726 -1.2124926   0.79801809 -0.98051059]
 [-1.36714726 -0.20723725 -1.28981993  0.75935442 -0.86451959]
 [ 0.95267275  1.26198209  0.25672675  0.79801809  0.91400909]
 [ 1.18465476 -0.43921926  1.72594609 -1.05783793  1.91926443]
 [-0.43921926  0.29539042 -0.36189192 -0.20723725 -1.13516526]]
```
```python
print("\nMean:", mean)
```
Output: ```Mean: 0.0```
```python
print("Standard Deviation:", std)
```
Output: ```Standard Deviation: 0.9999999999999999```
## **B. Cubes Divisible By 4 Problem**
Using NumPy, create the first 100 positive integers, cube every element, and reshape the result into a 10 × 10 **ndarray** named C. Thus, C begins with $1^3$ and ends with $100^3$. Use a Boolean condition on C to obtain every cubed value divisible by 4. Store the selected values in **div_by_4.** Preserve NumPy's normal row-major selection order. <br>
<br>
Functions used for this problem:
- **np.arange()**: **Creates an array** of evenly spaced values **within a specified numerical range**.
- **.reshape()**: **Changes the shape** or **dimension** of an array without changing the data.
- **Boolean Masking (`C % 4 == 0`)**: **Filters an array** using a condition to extract specific elements. For this example, it filters an array to find elements that are perfectly divisible by 4.
### **Code:**
```python
def cubes_divisible_by_4():
    C = (np.arange(1, 101) ** 3).reshape(10, 10)
    
    div_by_4 = C[C % 4 == 0]
    np.save('div_by_4.npy', div_by_4)
    
    return C.shape, div_by_4, div_by_4.size
```
#### **Test Cases:**
```python
shape, div_array, count = cubes_divisible_by_4()
```
```python
print("Shape of C:", shape)
```
Output: ```Shape of C: (10, 10)```
```python
print("div_by_4:", div_array)
```
Output: <br>
```
div_by_4: [      8      64     216     512    1000    1728    2744    4096    5832
    8000   10648   13824   17576   21952   27000   32768   39304   46656
   54872   64000   74088   85184   97336  110592  125000  140608  157464
  175616  195112  216000  238328  262144  287496  314432  343000  373248
  405224  438976  474552  512000  551368  592704  636056  681472  729000
  778688  830584  884736  941192 1000000]
```
```python
print("Number of selected elements:", count)
```
Output: ```Number of selected elements: 50```
## **C. Above-Mean Squares Problem**
Create a 6 × 6 ndarray named **S** containing the squares of the first 36 positive integers in increasing row-major order. Compute the mean of all elements of **S** and store it in **S_mean**. Then use Boolean filtering to select only the elements strictly greater than **S_mean**. Store these values in **above_mean**. <br>
<br>
Functions used for this problem:
- **np.mean()**: Computes the **mean** of the array elements
- **Boolean Masking (S > S_mean)**: **Filters** and selects only the elements that are greater than **S_mean**
### **Code:**
```python
def above_mean_squares():
    S = (np.arange(1, 37) ** 2).reshape(6, 6)

    S_mean = np.mean(S)
    above_mean = S[S > S_mean]
    np.save('above_mean.npy', above_mean)

    return S, S_mean, above_mean, above_mean.size
```
#### **Test Cases:**
```python
S, mean_val, above_array, count = above_mean_squares()
```
```python
print("S:\n", S)
```
Output: <br>
```
S:
 [[   1    4    9   16   25   36]
 [  49   64   81  100  121  144]
 [ 169  196  225  256  289  324]
 [ 361  400  441  484  529  576]
 [ 625  676  729  784  841  900]
 [ 961 1024 1089 1156 1225 1296]]
```
```python
print("\nS_mean:", mean_val)
```
Output: ```S_mean: 450.1666666666667```
```python
print("\nabove_mean:", above_array)
```
Output: <br>
```
above_mean: [ 484  529  576  625  676  729  784  841  900  961 1024 1089 1156 1225
 1296]
```
```python
print("\nNumber of selected elements:", count)
```
Output: ```Number of selected elements: 15```

To view and test the code:
- Download ```'Lacuna_PA - 2.ipynb'``` that is located in the repository
- Open the file via Jupyter Notebook
- Click on the file and click 'Run'

**README File Version History:**

```September 2, 2026``` - README.md output uploaded. <br>
