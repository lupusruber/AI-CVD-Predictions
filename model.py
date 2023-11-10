from torch import nn
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall
from copy import deepcopy
from data import create_train_data_loader
from data import scale_and_transfrom_data_for_training
from data import pca_data_loaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
torch.cuda.manual_seed(42)

accuracy_score = BinaryAccuracy().to(DEVICE)
precision = BinaryPrecision().to(DEVICE)
recall = BinaryRecall().to(DEVICE)


class NeuralNetworkModel(nn.Module):
    def __init__(self, input_features, hidden_units, output_features):
        super().__init__()
        self.layer_stack = nn.Sequential(
            # Input Layer
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.Sigmoid(),
            # Hidden Layers
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Sigmoid(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Sigmoid(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Sigmoid(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Sigmoid(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Sigmoid(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Sigmoid(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Sigmoid(),
            # Output Layer
            nn.Linear(in_features=hidden_units, out_features=output_features),
        )

    def forward(self, x):
        h = self.layer_stack(x)
        return h


def evaluate_performance(
    X_test, y_test, model_state_dict, input_features, hidden_units
):
    best_model = NeuralNetworkModel(
        input_features=input_features, hidden_units=hidden_units, output_features=1
    ).to(DEVICE)

    best_model.load_state_dict(model_state_dict)
    best_model.eval()
    with torch.inference_mode():
        test_logits = best_model(X_test).squeeze()
        y_test_pred = torch.round(torch.sigmoid(test_logits))
    print(
        "Model Accuracy Score is: ",
        100.0 * accuracy_score(preds=y_test_pred, target=y_test).item(),
    )
    print(
        "Model Precision is: ",
        100.0 * precision(preds=y_test_pred, target=y_test).item(),
    )
    print("Model Recall is: ", 100.0 * recall(preds=y_test_pred, target=y_test).item())


def train_and_eval_model(X_train, X_test, y_train, y_test):
    model = NeuralNetworkModel(
        input_features=11, hidden_units=128, output_features=1
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    number_of_epochs = 5000
    step_size = 10

    test_acc_list = []
    best_acc = 0.0
    model_state_dict = deepcopy(model.state_dict())

    for epoch in range(number_of_epochs):
        model.train()

        train_logits = model(X_train).squeeze()
        y_train_pred = torch.round(torch.sigmoid(train_logits))

        train_loss = loss_fn(train_logits, y_train)
        train_acc = 100.0 * accuracy_score(preds=y_train_pred, target=y_train).item()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test).squeeze()
            y_test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, y_test)
            test_acc = 100.0 * accuracy_score(preds=y_test_pred, target=y_test).item()

            if best_acc < test_acc:
                best_acc = test_acc
                model_state_dict = deepcopy(model.state_dict())

            if epoch % step_size == 0:
                test_acc_list.append(test_acc / 100)
                print(
                    f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train Acc.: {train_acc:.4f}% | Test loss: {test_loss:.4f} | Test Acc.: {test_acc:.4f}%"
                )

    return model_state_dict


def train_and_eval_batched_model():
    X_train, X_test, y_train, y_test = scale_and_transfrom_data_for_training()
    train_loader, test_loader = create_train_data_loader(
        X_train, X_test, y_train, y_test, 512
    )

    batched_model = NeuralNetworkModel(
        input_features=11, hidden_units=256, output_features=1
    ).to(DEVICE)

    batched_optimizer = torch.optim.Adam(batched_model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    model_state_dict = deepcopy(batched_model.state_dict())
    for epoch in range(501):
        batched_model.train()
        for X, y in train_loader:
            train_logits = batched_model(X).squeeze()
            train_loss = loss_fn(train_logits, y)

            batched_optimizer.zero_grad()
            train_loss.backward()
            batched_optimizer.step()

        acc = []
        batched_model.eval()
        with torch.inference_mode():
            for X, y in test_loader:
                test_logits = batched_model(X).squeeze()
                y_test_pred = torch.round(torch.sigmoid(test_logits))

                test_acc = 100.0 * accuracy_score(preds=y_test_pred, target=y).item()
                acc.append(test_acc)

            batched_test_acc = sum(acc) / len(acc)

            if best_acc < batched_test_acc:
                best_acc = batched_test_acc
                model_state_dict = deepcopy(batched_model.state_dict())

        if epoch % 50 == 0:
            print(f"Test accuracy after epoch {epoch}: {batched_test_acc:.4f}%")

    return model_state_dict


def train_and_eval_pca_model():
    train_loader, test_loader = pca_data_loaders()
    model = NeuralNetworkModel(
        input_features=5, hidden_units=128, output_features=1
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    model_state_dict = deepcopy(model.state_dict())
    best_acc = 0.0
    num_epochs = 501
    for epoch in range(num_epochs):
        model.train()
        for X, y in train_loader:
            train_logits = model(X).squeeze()
            y_train_pred = torch.round(torch.sigmoid(train_logits))

            train_loss = loss_fn(train_logits, y)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        acc = []
        model.eval()
        with torch.inference_mode():
            for X, y in test_loader:
                test_logits = model(X).squeeze()
                y_test_pred = torch.round(torch.sigmoid(test_logits))

                test_acc = 100.0 * accuracy_score(preds=y_test_pred, target=y).item()
                acc.append(test_acc)

            batched_test_acc = sum(acc) / len(acc)

            if best_acc < batched_test_acc:
                best_acc = batched_test_acc
                model_state_dict = deepcopy(model.state_dict())

        if epoch % 50 == 0:
            print(f"Test accuracy after epoch {epoch}: {sum(acc)/len(acc):.4f}%")

    return model_state_dict
