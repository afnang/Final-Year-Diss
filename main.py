import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import concurrent.futures
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from deap import base, creator, tools, algorithms
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import random
import time
import yfinance as yf

# Track start time for performance metrics
start_time = time.time()

# Lists to store training and validation losses
train_losses = []
val_losses = []


# Fetch historical stock data for NASDAQ (IXIC)
ticker_symbol = "^IXIC"
start_date = "1971-02-05"
end_date = "2024-03-08"

# Fetch data
data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Data manipulation as we are only interested in the date and the closing value of stock on that day for this experiment
data = data[['Close']].reset_index()
data['Date'] = pd.to_datetime(data['Date'])

# Use GPU acceleration if available
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Prepare the dataframe by shifting prices to create a dataset
def prepare_dataframe(df, n_steps):
    df = df.copy()
    df.set_index('Date', inplace=True)
    for i in range(1, n_steps+1):
        df[f'Closing Value(t-{i})'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

lookback = 20 # Lookback period (Can be adjusted)
shifted_df = prepare_dataframe(data, lookback) # Shifted dataframe
shifted_df_as_np = shifted_df.to_numpy() # Convert DataFrame to numpy array for processing
scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np) # Scale the data for neural network input


X = shifted_df_as_np[:, 1:]
y = shifted_df_as_np[:, 0]
X = np.flip(X, axis=1).copy()

# Splitting the dataset into training, validation, and test sets with a 70-15-15 distribution
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=(0.15 / 0.85), random_state=42)

# Reshaping (if necessary) and converting to tensors
X_train = X_train.reshape((-1, lookback, 1))
X_val = X_val.reshape((-1, lookback, 1))
X_test = X_test.reshape((-1, lookback, 1))

y_train = y_train.reshape((-1, 1))
y_val = y_val.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

# Convert arrays to tensors and upload to the GPU if available
X_train = torch.tensor(X_train).float().to(device)
X_val = torch.tensor(X_val).float().to(device)
X_test = torch.tensor(X_test).float().to(device)

y_train = torch.tensor(y_train).float().to(device)
y_val = torch.tensor(y_val).float().to(device)
y_test = torch.tensor(y_test).float().to(device)


# Dataset class
class TSD(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
# Initialise DataLoaders for batching during training and validation
train_dataset = TSD(X_train, y_train)
val_dataset = TSD(X_val, y_val)
test_dataset = TSD(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# LSTM Model 
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Taking the last time step output
        return out
    
model = LSTM(1, 512, 2).to(device) # Initialise model (Parameters can be adjusted)

# Main Training Loop 
def main_training_loop(learning_rate):
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.MSELoss().to(device)
    num_epochs = 300 #Set number of epochs for training (Can be adjusted)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            optimiser.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:  
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Output the training and validation loss for each epoch
        print(f'Epoch: {epoch+1}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')


# Genetic Algorithm Function
def evalOneMax(individual):
    lr = individual[0]
    loss = evaluate_model_with_lr(lr)
    print(f"Evaluating LR: {lr} with Loss: {loss}")  # Debugging
    return (-loss,)

def find_optimal_lr_with_ga():
    def checkBounds(min, max):
        def decorator(func):
            def wrapper(*args, **kargs):
                offspring = func(*args, **kargs)
                for child in offspring:
                    for i in range(len(child)):
                        if child[i] > max:
                            child[i] = max
                        elif child[i] < min:
                            child[i] = min
                return offspring
            return wrapper
        return decorator
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 1e-7, 1e-6)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.decorate("mate", checkBounds(1e-7, 1e-6))
    toolbox.decorate("mutate", checkBounds(1e-7, 1e-6))


    # Register the evaluation function
    toolbox.register("evaluate", evalOneMax)

    # Initialise the pool of processes
    # Allows for parallelisation
    pool = Pool()
    toolbox.register("map", pool.map)

    # Create an initial population
    pop = toolbox.population(n=5) # Can be adjusted to increase initial population
    print("Initial Population:", [ind[0] for ind in pop]) 
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=5, halloffame=hof, verbose=True) 


    print("Hall of Fame Individuals:")
    for individual in hof:
        print(f"LR: {individual[0]} | Fitness: {individual.fitness.values[0]}")

    print(f"Hall of Fame size: {len(hof)}")
    pool.close()
    if len(hof) > 0:
        print(f"Best individual: {hof[0]}")
    else:
        print("No individuals in Hall of Fame.")

    
    return hof[0][0]

# Evolutionary Strategy (ES) Function 
def es_optimise(lr_init=0.000001, sigma=0.001, lr_decay=0.9, iterations=5, num_workers=4): # Parameters can be adjusted
    best_lr = lr_init
    best_loss = float('inf')

    print("Starting ES optimisation")
    for iteration in range(iterations):
        print(f"Iteration {iteration + 1}/{iterations}")
        candidates = np.random.uniform(1e-7, 1e-6, 5)
        print("Candidates:", candidates)

        losses = []

        # Using ThreadPoolExecutor to parallelise evaluations
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_lr = {executor.submit(evaluate_model_with_lr, lr): lr for lr in candidates}
            for future in concurrent.futures.as_completed(future_to_lr):
                lr = future_to_lr[future]
                try:
                    loss = future.result()
                except Exception as e:
                    print(f"{lr} generated an exception: {e}")
                else:
                    print(f"Evaluating LR: {lr}, Loss: {loss}")
                    losses.append((lr, loss))

        # Finding the candidate with the best loss
        losses.sort(key=lambda x: x[1])
        if losses[0][1] < best_loss:
            best_loss = losses[0][1]
            best_lr = losses[0][0]
            print(f"New best LR: {best_lr}, Loss: {best_loss}")

        sigma *= lr_decay
        print(f"End of iteration {iteration + 1}, best LR: {best_lr}, best Loss: {best_loss}, next sigma: {sigma}")

    print(f"Final best LR: {best_lr}, Loss: {best_loss}")
    return best_lr





# Evaluate Model with Learning Rate Function 
def evaluate_model_with_lr(learning_rate, epochs=50):  # Number of epochs can be adjusted
    local_model = LSTM(1, 512, 2).to(device) 
    local_model.train()  # Ensure the model is in training mode
    optimiser = torch.optim.SGD(local_model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.MSELoss().to(device)

    for epoch in range(epochs):
        local_model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            optimiser.zero_grad()
            outputs = local_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

        local_model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = local_model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

    return avg_val_loss


# Define the loss function (criterion)
criterion = nn.MSELoss().to(device)

def evaluate_mse(test_loader, model, criterion, device='cpu'):
    model.eval()  # Set the model to evaluation mode
    test_loss_sum = 0
    count = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss_sum += loss.item() * data.size(0)
            count += data.size(0)

    avg_mse = test_loss_sum / count
    print(f'Average MSE on Test Set: {avg_mse}')


# User Input for Learning Rate Setting (unchanged)
def get_user_input():
    print("Select the method to set the learning rate:")
    print("1: Use default value (0.000001)")
    print("2: Use Genetic Algorithm (GA) with DEAP to find an optimal value")
    print("3: Use Evolutionary Strategy (ES) to find an optimal value")
    choice = input("Enter your choice (1/2/3): ")
    return int(choice)

if __name__ == '__main__':
    user_choice = get_user_input()

    if user_choice == 1:
        selected_lr = 0.000001
    elif user_choice == 2:
        selected_lr = find_optimal_lr_with_ga()
    elif user_choice == 3:
        selected_lr = es_optimise()

    print(f"Selected Learning Rate: {selected_lr}")
    main_training_loop(selected_lr)
    evaluate_mse(test_loader, model, criterion, device)

    # Plotting the results
    with torch.no_grad():
        model.eval()
        predicted = model(X_train).cpu().numpy()
    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"Total runtime of the program: {total_runtime} seconds.")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss at LR: {selected_lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Logarithmic)')
    plt.yscale('log')  # Set y-axis scale to logarithmic
    plt.legend()
    plt.grid(True)
    plt.show()

    

    