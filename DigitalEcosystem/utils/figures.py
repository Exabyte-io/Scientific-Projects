import matplotlib
import numpy as np
import seaborn as sns
import sklearn.metrics
from matplotlib import pyplot as plt


def plot_roc(x, y, label, classifier):
        plt.rcParams['figure.figsize'] = [10,10]
        probabilities = classifier.predict_proba(x)[:,1]

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

        plt.plot([0,1], [0,1], c='k')

        plt.title(f"{label} Set ROC curve, AUC={roc_auc}")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.tight_layout()
        plt.savefig(f"{label}_Set_ROC.png")
        plt.show()
        plt.close()


def draw_confusion_matrix(x, y, label, classifier, cutoff=0.5):
    plt.rcParams['figure.facecolor'] = 'white'
    sklearn.metrics.ConfusionMatrixDisplay(
        sklearn.metrics.confusion_matrix(
            y_true=y,
            y_pred=classifier.predict_proba(x)[:,1]>cutoff,
        )
    ).plot(cmap="Blues")
    plt.title(f"{label} Set Confusion Matrix at cutoff {cutoff}")
    plt.xticks([0,1], labels=["Nonmetal", "Metal"])
    plt.yticks([0,1], labels=["Nonmetal", "Metal"])
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

def create_multi_parity_plot(ytrue, ypred, source_df, is_train):
    tpot_mape = np.round(sklearn.metrics.mean_absolute_percentage_error(y_true=unscale(ytrue), y_pred=unscale(ypred)),2)
    r1_1t_mape = np.round(sklearn.metrics.mean_absolute_percentage_error(y_true=unscale(ytrue), y_pred=unscale(source_df["r1_1term"])),2)
    r1_2t_mape = np.round(sklearn.metrics.mean_absolute_percentage_error(y_true=unscale(ytrue), y_pred=unscale(source_df["r1_2term"])),2)
    r2_1t_mape = np.round(sklearn.metrics.mean_absolute_percentage_error(y_true=unscale(ytrue), y_pred=unscale(source_df["r2_1term"])),2)

    plt.rcParams["figure.dpi"]=200
    plt.scatter(x=unscale(train_pred_y), y=unscale(ytrue), color="black", alpha=0.9, marker="+", label=f"TPOT, 108 Terms, MAPE={tpot_mape}")
    plt.scatter(x=unscale(source_df["r1_1term"]), y=unscale(ytrue), marker="v", color="red",alpha=0.5, label=f"Rung 1, 1-Term, MAPE={r1_1t_mape}")
    plt.scatter(x=unscale(source_df["r1_2term"]), y=unscale(ytrue), marker="^", color="green", alpha=0.5, label=f"Rung 1, 2-Term, MAPE={r1_2t_mape}")
    plt.scatter(x=unscale(source_df["r2_1term"]), y=unscale(ytrue), marker="s", color="blue", alpha=0.5, label=f"Rung 2, 1-term, MAPE={r2_1t_mape}")
    plt.plot([45, 280], [45, 280], color="black", linestyle="--", label="Parity")

    if is_train:
        plt.title("Training Set (80% of Dataset)")
    else:
        plt.title("Testing Set (20% Holdout)")
    plt.xlabel("Predicted (Å^3 / Formula Unit)")
    plt.ylabel("Actual Volume (Å^3 / Formula Unit)")
    plt.legend(prop={"size": 8})
    plt.show()

def create_parity_plot(train_yhat, test_yhat):
    plt.scatter(x=unscale(train_yhat), y=unscale(train_y), label="Train")
    plt.scatter(x=unscale(test_yhat), y=unscale(test_y), label="Test")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.legend()
    plt.show()
