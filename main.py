import torch
import torch.optim as optim
import pickle
from Model import HierachicalAttentionNetwork as HAN
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

if __name__ == '__main__':
    word_emmbed_dim = 200
    word_gru_size = sentence_gru_size = 50
    label_class = 5
    
    with open('Data/X.pickle','rb') as file:
        X = pickle.load(file)
    with open('Data/Y.pickle','rb') as file:
        Y = pickle.load(file)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    model = HAN(word_emmbed_dim, word_gru_size, sentence_gru_size, label_class)
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.NLLLoss()

    num_epochs = 10

    losses = []
    for i in range(num_epochs):
        running_loss = 0.0
        for input, target in zip(X_train, Y_train):
            output = model(input)

            target = torch.tensor([target]) 
            loss = criterion(torch.log(output).unsqueeze(0),target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

        losses.append(running_loss)
        print(f'epoch: {i}, loss: {running_loss}')

    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.show()

    correct = 0
    with torch.no_grad():
        for input,target in zip(X_test, Y_test):
            Y_pred = model(input)
            correct += (torch.argmax(Y_pred) == torch.argmax(target)).item()

    print('Accuracy of the network on test set: %d %%' % (100 * correct / len(X_test)))
    