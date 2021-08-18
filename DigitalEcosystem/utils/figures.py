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


def draw_confusion_matrix(x,y,label,classifier, cutoff=0.5):
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
