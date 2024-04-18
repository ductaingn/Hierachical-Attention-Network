import torch
import torch.optim as optim
import pickle
from Model import HierachicalAttentionNetwork as HAN
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    word_emmbed_dim = 200
    word_gru_size = sentence_gru_size = 50
    label_class = 5
    
    model = HAN(word_emmbed_dim, word_gru_size, sentence_gru_size, label_class)
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.NLLLoss()

    num_epochs = 1000

    with open('Data/X.pickle','rb') as file:
        X = pickle.load(file)
    with open('Data/Y.pickle','rb') as file:
        Y = pickle.load(file)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    for i in range(num_epochs):
        count = 0
        for input, target in zip(X_train, Y_train):
            output = model(input)
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            count+=1
    
    