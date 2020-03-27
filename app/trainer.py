import torch


def train(model, iterator, optimizer, criterion, metric):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        # retrieve text and no. of words
        text, text_lengths = batch.text

        predictions = model(text, text_lengths).squeeze()
        loss = criterion(predictions, batch.label)

        # compute the metric
        acc = metric(predictions, batch.label)

        loss.backward()
        optimizer.step()

        # loss and accuracy
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    # TODO change metric to f1
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, metric):
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # deactivating dropout layers
    model.eval()

    # deactivates autograd
    with torch.no_grad():
        for batch in iterator:
            # retrieve text and no. of words
            text, text_lengths = batch.text

            # convert to 1d tensor
            predictions = model(text, text_lengths).squeeze()

            # compute loss and accuracy
            loss = criterion(predictions, batch.label)
            acc = metric(predictions, batch.label)

            # keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    # TODO change metric to f1
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def train_model(model, train_iter, test_iter, optimizer, criterion, metric):
    N_EPOCHS = 5
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        # train the model
        train_loss, train_acc = train(model, train_iter, optimizer, criterion, metric)

        # evaluate the model
        valid_loss, valid_acc = evaluate(model, test_iter, criterion, metric)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights.pt')

        # TODO add f1 calculation
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
