import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from data import get_data, get_correlation_matrix, pca_for_data_analysis

data_frame = get_data()


def create_gender_pairplot():
    columns_to_drop = ["smoke", "alco", "active", "cardio"]
    data_frame_for_viz = data_frame.drop(columns=columns_to_drop, axis=1)

    sns.pairplot(data=data_frame_for_viz, hue="gender")
    plt.show()


def create_cardio_pariplot():
    columns_to_drop = [
        "smoke",
        "alco",
        "active",
    ]
    data_frame_for_viz = data_frame.drop(columns=columns_to_drop, axis=1)

    sns.pairplot(data=data_frame_for_viz, hue="cardio")
    plt.show()


def create_heatmap():
    correlation_matrix = get_correlation_matrix()
    ax = sns.heatmap(correlation_matrix, annot=False, linewidth=1.0)
    plt.show()


def epochs_vs_acc_graph(number_of_epochs, step_size, test_acc_list):
    x = list(range(0, number_of_epochs, step_size))
    sns.lineplot(x=x, y=test_acc_list, color="red", linewidth=1)
    ax = plt.gca()
    ax.set_xlabel("epochs")
    ax.set_ylabel("accuracy")
    plt.show()


def pca_data_visualization():
    pca_data = pca_for_data_analysis()
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=data_frame["cardio"])
    plt.show()


def pca_box_plot():
    pca_data = pca_for_data_analysis()
    plt.boxplot(pca_data)
    plt.xlabel("Principal Components")
    plt.ylabel("Values")
    plt.title("Box Plot for PCA Data")
    plt.show()
