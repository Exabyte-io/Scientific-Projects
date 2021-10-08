import functools
import matplotlib
import numpy as np
import seaborn as sns
import sklearn.metrics
from matplotlib import pyplot as plt


def plot_roc(x, y, label, classifier):
    plt.rcParams['figure.figsize'] = [10, 10]
    probabilities = classifier.predict_proba(x)[:, 1]

    # ROC curve function in sklearn prefers the positive class
    false_positive_rate, true_positive_rate, thresholds = sklearn.metrics.roc_curve(y, probabilities,
                                                                                    pos_label=1)
    thresholds[0] -= 1  # Sklearn arbitrarily adds 1 to the first threshold
    roc_auc = np.round(sklearn.metrics.auc(false_positive_rate, true_positive_rate), 3)

    # Plot the curve
    fig, ax = plt.subplots()
    points = np.array([false_positive_rate, true_positive_rate]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(thresholds.min(), thresholds.max())
    lc = matplotlib.collections.LineCollection(segments, cmap='jet', norm=norm, linewidths=2)
    lc.set_array(thresholds)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax).set_label('Threshold')

    # Padding to ensure we see the line
    ax.margins(0.01)

    fig.patch.set_facecolor('white')

    plt.plot([0, 1], [0, 1], c='k')

    plt.title(f"{label} Set ROC curve, AUC={roc_auc}")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(f"{label}_Set_ROC.png")
    plt.show()
    plt.close()


def plot_multi_roc(x, y, dataset_label, classifier, custom_labels=None):
    plt.rcParams['figure.figsize'] = [10, 10]

    classes = set(y)
    fig, ax = plt.subplots()

    for class_label in classes:
        probabilities = classifier.predict_proba(x)[:, class_label]

        # ROC curve function in sklearn prefers the positive class
        false_positive_rate, true_positive_rate, thresholds = sklearn.metrics.roc_curve(y, probabilities,
                                                                                        pos_label=class_label)
        roc_auc = np.round(sklearn.metrics.auc(false_positive_rate, true_positive_rate), 3)

        if custom_labels is None:
            label = f"Class {class_label}"
        else:
            label = custom_labels[class_label]
        ax.plot(false_positive_rate, true_positive_rate, label=f"{label}, AUC ROC={roc_auc}")

    # Padding to ensure we see the line
    ax.margins(0.01)
    ax.legend()
    fig.patch.set_facecolor('white')
    plt.plot([0, 1], [0, 1], c='k')
    plt.title(f"{dataset_label} Set ROC curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(f"{dataset_label}_Set_ROC.png")
    plt.show()
    plt.close()


def draw_confusion_matrix(x, y, label, classifier, cutoff=0.5):
    plt.rcParams['figure.facecolor'] = 'white'
    sklearn.metrics.ConfusionMatrixDisplay(
        sklearn.metrics.confusion_matrix(
            y_true=y,
            y_pred=classifier.predict_proba(x)[:, 1] > cutoff,
        )
    ).plot(cmap="Blues")
    plt.title(f"{label} Set Confusion Matrix at cutoff {cutoff}")
    plt.xticks([0, 1], labels=["Nonmetal", "Metal"])
    plt.yticks([0, 1], labels=["Nonmetal", "Metal"])
    plt.gca().xaxis.tick_top()
    plt.savefig(f"{label}_set_confusion_matrix.png")
    plt.show()
    plt.close()

def draw_confusion_matrix_custom_classnames(x, y, label, classnames, classifier):
    plt.rcParams['figure.facecolor'] = 'white'
    sklearn.metrics.ConfusionMatrixDisplay(
        sklearn.metrics.confusion_matrix(
            y_true=y,
            y_pred=classifier.predict(x),
        )
    ).plot(cmap="Blues")
    plt.title(f"{label} Set Confusion Matrix")
    plt.xticks(range(len(classnames)), labels=classnames)
    plt.yticks(range(len(classnames)), labels=classnames)
    plt.gca().xaxis.tick_top()
    plt.savefig(f"{label}_set_confusion_matrix.png")
    plt.show()
    plt.close()


def save_train_test_histplot(train_df, test_df, title, filename, column, stat, bins):
    sns.histplot(train_df, x=column, label="Train", color="#AAAAFF", stat=stat, bins=bins)
    sns.histplot(test_df, x=column, label="Test", color="#FFAAAA", stat=stat, bins=bins)
    plt.legend()
    plt.title(title)
    plt.savefig(filename)


def save_parity_plot_with_raw_values(train_pred_y, test_pred_y, train_y, test_y, filename):
    plt.scatter(x=train_y, y=train_pred_y, label="Train")
    plt.scatter(x=test_y, y=test_pred_y, label="Test")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.savefig(filename)
    plt.legend()
    plt.show()


def save_parity_plot(train_x, test_x, train_y, test_y, regression_model, label, filename):
    train_pred_y = regression_model.predict(train_x)
    test_pred_y = regression_model.predict(test_x)

    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
    plt.rcParams["figure.figsize"] = (15, 15)
    plt.rcParams["font.size"] = 16

    plt.scatter(x=train_y, y=train_pred_y, label="Train Set")
    plt.scatter(x=test_y, y=test_pred_y, label="Test Set")

    min_xy = min(min(train_y), min(test_y), min(test_pred_y), min(train_pred_y))
    max_xy = max(max(train_y), max(test_y), max(test_pred_y), max(train_pred_y))

    plt.plot([min_xy, max_xy], [min_xy, max_xy], label="Parity")
    plt.ylabel(f"{label} (Predicted)")
    plt.xlabel(f"{label} (Dataset)")
    plt.legend()
    plt.savefig(filename)
    plt.show()
    plt.close()


def create_multi_parity_plot(ytrue, series_to_plot, markers, colors, labels, alphas, is_train):
    plt.rcParams["figure.dpi"] = 200

    parity_min = min(ytrue)
    parity_max= max(ytrue)

    for series, marker, color, label, alpha in zip(series_to_plot, markers, colors, labels, alphas):
        plt.scatter(x=series, y=ytrue, color=color, marker=marker, alpha=alpha, label=label)
        parity_min = min(parity_min, min(series))
        parity_max = max(parity_max, max(series))

    plt.plot([parity_min, parity_max], [parity_min, parity_max], color="black", linestyle="--", label="Parity")

    if is_train:
        plt.title("Training Set (80% of Dataset)")
    else:
        plt.title("Testing Set (20% Holdout)")
    plt.xlabel("Predicted (Å^3 / Formula Unit)")
    plt.ylabel("Actual Volume (Å^3 / Formula Unit)")
    plt.legend(prop={"size": 8})
    plt.show()


def save_parity_plot_publication_quality(train_y_true,
                                         train_y_pred,
                                         test_y_true,
                                         test_y_pred,
                                         axis_label,
                                         filename=None,
                                         axis_limits=None,
                                         title=None):
    plt.scatter(x=train_y_true, y=train_y_pred, label="Train Set")
    plt.scatter(x=test_y_true, y=test_y_pred, label="Test Set")

    if axis_limits is None:
        min_xy = min(min(train_y_true), min(test_y_true), min(test_y_pred), min(train_y_pred))
        max_xy = max(max(train_y_true), max(test_y_true), max(test_y_pred), max(train_y_pred))
    else:
        min_xy = axis_limits
        max_xy = axis_limits


    plt.plot([min_xy, max_xy], [min_xy, max_xy], label="Parity")

    plt.ylabel(f"{axis_label} (Predicted)")
    plt.xlabel(f"{axis_label} (Dataset)")
    if title:
        plt.title(title)
    plt.legend()

    if filename:
        plt.savefig(filename)

    plt.show()
    plt.close()
