# Import all necessary Modules or Libraries
import numpy as np 
import matplotlib.pyplot as plt 

# Function to calculate coefficients using closed-form solution
def calculate_coefficient(a, Y):
    # Create the design matrix X
    X = np.vstack((np.ones(len(a)), a)).T
    
    # Calculate coefficients using the closed-form solution
    betas = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))
    b0 = betas[0]
    b1 = betas[1]
    return b0, b1

# Function to calculate mean squared error
def calculate_mse(y, ycap):
    return np.mean((y - ycap)**2)

# Generate random data
x_vals = np.arange(-5, 5, 0.01)
y_vals = 2 * x_vals - 3 + np.random.normal(0, 5)
dataset = list(zip(x_vals, y_vals))
np.random.shuffle(dataset)

# Split data into training and testing sets
datatrain = dataset[:int(0.8*len(dataset))]
datatest = dataset[int(0.8*len(dataset)):]

# Extract x and y values for training and testing sets
x_train = np.array([i[0] for i in datatrain]) 
y_train = np.array([i[1] for i in datatrain]) 
x_test = np.array([i[0] for i in datatest]) 
y_test = np.array([i[1] for i in datatest]) 

# Calculate coefficients using closed-form solution
eb0, eb1 = calculate_coefficient(x_train, y_train)

# Calculate predictions and error using closed-form solution
yclosed = eb0 + eb1*x_train 
errorclosed = calculate_mse(y_train, yclosed) 

# Initialize random coefficients
initial_b0 = np.random.normal(0, 1) 
initial_b1 = np.random.normal(0, 1) 

# Calculate initial error for training and testing sets
initial_train_error = calculate_mse(y_train[0], initial_b0 + initial_b1*x_train[0]) 
initial_test_error = calculate_mse(y_test[0], initial_b0 + initial_b1*x_test[0]) 

# Define learning rate
learning_rate_list = [0.001] 

# Iterate over learning rates
for i in range(len(learning_rate_list)): 
    lr = learning_rate_list[i] 
    error = initial_train_error
    b0 = initial_b0 
    b1 = initial_b1 
    epoch = 0
    epoch_list = [0]
    train_error_list = [initial_train_error] 
    test_error_list = [initial_test_error]
    converged = False
    
    # Mini-batch gradient descent
    while not converged: 
        # Split data into mini-batches
        xnew = [x_train[i:i+50] for i in range(0, len(x_train), 50)] 
        ynew = [y_train[i:i+50] for i in range(0, len(y_train), 50)] 
        
        # Iterate over mini-batches
        for x_train, y_train in zip(xnew, ynew): 
            ycap = b0 + b1*x_train
            
            # Calculate gradients
            gradb0 = -2*np.mean(y_train - ycap) 
            gradb1 = -2*np.mean((y_train - ycap)*x_train)
            
            # Update coefficients
            b0 = b0 - lr*gradb0
            b1 = b1 - lr*gradb1
                            
            # Calculate error for training and testing sets
            new_train_error = calculate_mse(y_train, b0 + b1*x_train)
            new_test_error = calculate_mse(y_test, b0 + b1*x_test)   
                
            # Append errors to error lists  
            train_error_list.append(new_train_error)  
            test_error_list.append(new_test_error)  
            
            # Update epoch count
            epoch += 1
            epoch_list.append(epoch)
            
        # Check for convergence
        if abs(error - new_train_error) < 10e-6:
            converged = True
        else:
            error = new_train_error
            
    # Convert epoch list to represent number of epochs instead of iterations
    epoch_list = [i//len(x_train) for i in epoch_list]     
    
    # Print results
    print("Using The Closed Form :", "b0:", eb0, "b1:", eb1, "error:", errorclosed)
    print("Using MiniBatch Gradient Descent :", "b0:", b0, "b1:", b1, "error:", error, "epoch:", epoch//(50), "learning rate:", lr)
    
    # Plot errors over epochs
    plt.plot(epoch_list, train_error_list, label = "Bias") 
    plt.plot(epoch_list, test_error_list, label = "Variance") 

# Add labels and title to the plot
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.xlim(0, 25)
plt.title('MiniBatch Gradient Descent')

# Add additional text to the plot
plt.figtext(0.5, 0.01, "In the above graph, the Mean Squared Error of Training data and testing data is plotted with respect to the number of minibatch iterations executed", ha="center", fontsize=10, bbox={"facecolor":"brown", "alpha":0.5, "pad":5})

# Display the plot
plt.show()
# import numpy as np
# import matplotlib.pyplot as plt

# # Generating random 1000 values to be appended in x
# x = []
# # array to store calculated y values
# y = []
# # x matrix to get squared values for calculation of beta values
# x1 = [[1] * 100, []]
# # iterating each of the 100 values to calculate actual output
# for i in range(-50, 50):
#     x_val = i / 100
#     x.append(x_val)
#     x1[1].append(i)
#     n = np.random.normal(0, 5)
#     y_val = 2 * x_val - 3 + n
#     y.append(y_val)

# # Splitting the data into training and testing sets (80-20 split)
# split_index = int(len(x) * 0.8)
# x_train, x_test = x[:split_index], x[split_index:]
# y_train, y_test = y[:split_index], y[split_index:]

# # Normalizing the data
# x_train_mean, x_train_std = np.mean(x_train), np.std(x_train)
# x_train = (x_train - x_train_mean) / x_train_std
# x_test = (x_test - x_train_mean) / x_train_std

# # Normalizing the target variable
# y_train_mean, y_train_std = np.mean(y_train), np.std(y_train)
# y_train = (y_train - y_train_mean) / y_train_std
# y_test = (y_test - y_train_mean) / y_train_std

# # getting transpose of matrix to get closed form values
# X1 = np.transpose(x1)
# Y1 = np.transpose(y)

# # Calculate coefficients using matrix multiplication
# b = np.matmul(np.linalg.inv(np.matmul(x1, X1)), np.matmul(x1, Y1))
# B0 = b[0]
# B1 = b[1]

# # Gradient Descent processing
# b0f = np.random.normal(0, 1)
# b1f = np.random.normal(0, 1)

# # calculating error function
# errorf = np.mean((y - (b0f + b1f * np.array(x))) ** 2)
# lr = 0.001  # Adjusted learning rate

# # finding error values for initial b0 and b1 values
# error = errorf
# b0 = b0f
# b1 = b1f
# epoch = 0

# # initializing epoch matrix
# Epoch = [0]
# E_train = [errorf]
# E_test = [errorf]

# # array to store gradient b0 and b1 value
# Gb0 = []
# Gb1 = []

# out = False

# # splitting the dataset into minibatches
# num_minibatches = 20
# minibatch_size = 50
# minibatches = np.array_split(x_train, num_minibatches)

# # updating epoch values until the error is close to near 0 values
# while not out:
#     for minibatch in minibatches:
#         # calculating gradients for the current minibatch
#         grad_b0 = 0
#         grad_b1 = 0
#         for i in range(len(minibatch)):
#             grad_b0 += -2 * ((y_train[i] - b0 + b1 * minibatch[i]) * minibatch[i])
#             grad_b1 += -2 * ((y_train[i] - b0 + b1 * minibatch[i]) * minibatch[i])
#         grad_b0 /= len(minibatch)
#         grad_b1 /= len(minibatch)
        
#         # updating new b0 and b1 values
#         b0 -= lr * grad_b0
#         b1 -= lr * grad_b1
        
#         # computing error for new b0 and b1 values for training set
#         ne_train = (y_train - (b0 + b1 * np.array(x_train))) ** 2
#         new_error_train = np.mean(ne_train)
        
#         # computing error for new b0 and b1 values for testing set
#         ne_test = (y_test - (b0 + b1 * np.array(x_test))) ** 2
#         new_error_test = np.mean(ne_test)
        
#         epoch += 1
#         # appending epoch values
#         Epoch.append(epoch)
#         E_train.append(new_error_train)
#         E_test.append(new_error_test)
#         Gb0.append(b0)
#         Gb1.append(b1)
        
#         # checking stop condition
#         if abs(error - new_error_train) < 10e-6:
#             out = True
#         else:
#             error = new_error_train

# print("Closed Form: b0:", B0, "b1:", B1, "error:", errorf)
# print("Minibatch Gradient Descent: b0:", b0, "b1:", b1, "error:", error, "epoch:", epoch)

# # Plotting the error convergence
# plt.figure(figsize=(8, 4))
# plt.plot(Epoch, E_train, label="Training MSE")
# plt.plot(Epoch, E_test, label="Testing MSE")
# plt.xlabel('Number of Epoch Cycles')
# plt.ylabel('Mean Squared Error')
# plt.xlim(5,None)
# plt.title('Minibatch Gradient Descent')
# plt.legend()
# plt.figtext(0.5, 0.01, "In the above graph the Mean Square Error is plotted with respect to the number of Epoch cycle is executed", ha="center", fontsize=10, bbox={"facecolor": "brown", "alpha": 0.5, "pad": 5})
# plt.show()