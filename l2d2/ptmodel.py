import datetime
import queue

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import scoped_time
from . import ingest
from . import sampler

import xtrie


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.fc1 = nn.Linear(8 * 15, 512)
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.flatten = nn.Flatten()
        self.last = nn.Linear(288, 9)
        #self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.last(x)
        return x

# initialize the NN
model = Net()
model.to('cuda')
print(model)

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
n_epochs = 32
batch_size = 128

model.train()

def train(sampler):
    for epoch in range(n_epochs):
        epoch_start = datetime.datetime.now()
        train_loss = 0.0
        
        while True:
            mb = sampler.q.get()
            if mb is None:
                break
            floats = torch.from_numpy(mb.floats).cuda()
            lengths = torch.from_numpy(mb.lengths).type(torch.long).cuda()
            #print('i:', floats)
            #print('l:', lengths)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(floats)
            # calculate the loss
            loss = criterion(output, lengths)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*lengths.size(0)

        # print training statistics 
        # calculate average loss over an epoch
        train_loss = train_loss/(sampler.i*batch_size)

        steps_per_sec = sampler.i/(datetime.datetime.now()-epoch_start).total_seconds()
        print('Epoch: {} \tTraining Loss: {:.6f}\t{:.2f} steps/s'.format(
            epoch+1, 
            train_loss, steps_per_sec
            ))
        sampler.restart()


def do_eval(sampler):
    sampler.restart()

    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval() # prep model for *evaluation*

    COUNT = 1024
    for i in range(COUNT):
        mb = sampler.q.get()
        data = torch.from_numpy(mb.floats).cuda()
        target = torch.from_numpy(mb.lengths).type(torch.long).cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss/(COUNT*batch_size)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (i,))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))



def main(do_train: bool):
    global model
    xtopts = xtrie.mk_opts()
    with scoped_time('loading x86 data'):
        data = ingest.load_state('/tmp/x86.state.big', xtopts)
    q = queue.Queue(maxsize=64)
    s = sampler.SamplerThread(q, data, batch_size)
    s.start()

    if do_train:
        try:
            start = datetime.datetime.now()
            train(s)
            end = datetime.datetime.now()
        except Exception as e:
            print('... cancelling due to exception:', e)
            s.cancel()
            s.join()
            raise

        print('... done training in', end-start)
        torch.save(model, '/tmp/model')
    else:
        model = torch.load('/tmp/model')

    do_eval(s)

    s.cancel()
    s.join()


if __name__ == '__main__':
    main(True)
