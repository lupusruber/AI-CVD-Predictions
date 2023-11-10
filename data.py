import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_data():
    data_frame = pd.read_csv(
        r"/home/pmalinov/AI-CVD-Predictions/cardio_train.csv", sep=";"
    )
    data_frame.drop("id", axis=1, inplace=True)

    age_array = data_frame["age"].values
    age_array = np.floor(age_array / 365)
    data_frame["age"] = np.array(age_array, dtype=int)

    return data_frame


def get_correlation_matrix():
    data_frame = get_data()
    return data_frame.corr().copy()


def get_values_from_data():
    data_frame = get_data()
    return data_frame.values


def exploratory_data_analysis():
    data_frame = get_data()

    gender_df_values = data_frame["gender"].values
    gender_array = ["male" if entry == 2 else "female" for entry in gender_df_values]
    data_frame["gender"] = gender_array

    print(data_frame.head())

    print(data_frame.groupby("gender").size())

    print(data_frame.groupby("cardio").size())


def scale_and_transfrom_data_for_training():
    values_matrix = get_values_from_data()
    X_tensor = torch.from_numpy(values_matrix[:, :-1]).to(DEVICE, dtype=torch.float32)
    y_tensor = torch.from_numpy(values_matrix[:, -1]).to(DEVICE, dtype=torch.float32)

    scaler_train = StandardScaler()
    scaler_test = StandardScaler()

    X_train_not_scaled, X_test_not_scaled, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )

    X_tensor_scaled = scaler_train.fit_transform(X_train_not_scaled.cpu())
    X_train = torch.from_numpy(X_tensor_scaled).to(DEVICE, dtype=torch.float32)

    X_tensor_scaled = scaler_test.fit_transform(X_test_not_scaled.cpu())
    X_test = torch.from_numpy(X_tensor_scaled).to(DEVICE, dtype=torch.float32)

    return X_train, X_test, y_train, y_test


def create_train_data_loader(train_X, test_X, train_y, test_y, BATCH_SIZE):
    train_data_set_new = torch.utils.data.TensorDataset(train_X, train_y)
    train_loader = torch.utils.data.DataLoader(
        train_data_set_new, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    test_data_set_new = torch.utils.data.TensorDataset(test_X, test_y)
    test_loader = torch.utils.data.DataLoader(
        test_data_set_new, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    return train_loader, test_loader


def pca_data_loaders():
    values_matrix = get_values_from_data()
    X = values_matrix[:, :-1]
    Y = values_matrix[:, -1]

    pca = PCA(n_components=5)
    X = pca.fit_transform(X)

    X_tensor = torch.from_numpy(X).to(DEVICE, dtype=torch.float32)
    y_tensor = torch.from_numpy(Y).to(DEVICE, dtype=torch.float32)

    scaler_train = StandardScaler()
    scaler_test = StandardScaler()

    X_train_not_scaled, X_test_not_scaled, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )

    X_tensor_scaled = scaler_train.fit_transform(X_train_not_scaled.cpu())

    X_train = torch.from_numpy(X_tensor_scaled).to(DEVICE, dtype=torch.float32)

    X_tensor_scaled = scaler_test.fit_transform(X_test_not_scaled.cpu())
    X_test = torch.from_numpy(X_tensor_scaled).to(DEVICE, dtype=torch.float32)

    train_loader, test_loader = create_train_data_loader(
        X_train, X_test, y_train, y_test, 512
    )

    return train_loader, test_loader


def pca_for_data_analysis():
    data_frame = get_data()
    gender_df_values = data_frame["gender"].values
    gender_array = [2 if entry == "male" else 1 for entry in gender_df_values]
    data_frame["gender"] = gender_array

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_frame)

    n_components = 2
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)

    new_x = pd.DataFrame(data=pca_data, columns=["PC1", "PC2"])

    return new_x
