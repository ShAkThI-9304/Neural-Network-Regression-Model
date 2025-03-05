# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

![image](https://github.com/user-attachments/assets/3745ebb7-5918-4898-a14c-4773d04b0a4e)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:Sakthivel B
### Register Number:212222040141
```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.history = {'loss': []}
        self.linear1 = nn.Linear(1, 12)
        self.linear2 = nn.Linear(12, 10)
        self.linear3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

  def forward(self,x):
    x = self.relu(self.linear1(x))
    x = self.relu(self.linear2(x))
    x = self.linear3(x)
    return x



# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)



def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
  for epoch in range(epochs):
    optimizer.zero_grad()
    loss = criterion(ai_brain(X_train), y_train)
    loss.backward()
    optimizer.step()

    ai_brain.history['loss'].append(loss.item())
    if epoch % 200 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')



```
## Dataset Information

![image](https://github.com/user-attachments/assets/b035b6de-dc89-4527-99ea-7428a4041b38)


## OUTPUT


### Training Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/57db2573-ab43-4b31-ab13-37e5b82b2b64)


### New Sample Data Prediction

Include your sample input and output here

![image](https://github.com/user-attachments/assets/ffe03c30-cb09-4415-bd8e-662f33024a11)

![image](https://github.com/user-attachments/assets/b249a63e-08c2-437a-aa18-85887e2e0949)



## RESULT

Include your result here
