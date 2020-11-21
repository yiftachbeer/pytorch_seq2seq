import math
import time

import torch


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class Model:

    def __init__(self, network, optimizer, criterion):
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, iterator, clip):
        self.network.train()

        epoch_loss = 0

        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            self.optimizer.zero_grad()

            output = self.network(src, trg)

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = self.criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), clip)
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def evaluate(self, iterator):
        self.network.eval()

        epoch_loss = 0

        with torch.no_grad():

            for i, batch in enumerate(iterator):

                src = batch.src
                trg = batch.trg

                output = self.network(src, trg, 0) #turn off teacher forcing

                #trg = [trg len, batch size]
                #output = [trg len, batch size, output dim]

                output_dim = output.shape[-1]

                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)

                #trg = [(trg len - 1) * batch size]
                #output = [(trg len - 1) * batch size, output dim]

                loss = self.criterion(output, trg)

                epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def predict(self, sentence_tensor):
        self.network.eval()

        with torch.no_grad():
            output = self.network(sentence_tensor, teacher_forcing_ratio=0)  #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

        return output

    def train_epochs(self, train_iterator, valid_iterator):
        N_EPOCHS = 10
        CLIP = 1

        best_valid_loss = float('inf')

        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss = self.train(train_iterator, CLIP)
            valid_loss = self.evaluate(valid_iterator)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.network.state_dict(), 'weights/tut1-model.pt')

            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')