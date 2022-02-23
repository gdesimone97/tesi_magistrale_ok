import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path

class PlotManager:
    def __init__(self, exp_dir, min_snr, max_snr, snr_step):
        self.exp_dir = exp_dir
        self.min_snr, self.max_snr, self.snr_step = min_snr, max_snr, snr_step
        self.fig_dir = Path(self.exp_dir).joinpath("res/fig")
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.init_tags()
        self.scalars = self.get_scalars()

    def init_tags(self):
        acc_list = ["accuracy_mean", "bal_acc_mean", "accuracy_snr_{}dB", "bal_acc_snr_{}dB"]
        self.acc_list = set()
        for e in acc_list:
            for snr in range(self.min_snr, self.max_snr+self.snr_step, self.snr_step):
                element = e.format(snr)
                self.acc_list.add(element)
        self.acc_list = list(self.acc_list)

    def get_scalars(self):
        scalars = {}
        event_acc = event_accumulator.EventAccumulator(self.exp_dir)
        event_acc.Reload()
        tags: list = event_acc.Tags()["scalars"]
        tags.remove("hp_metric"), tags.remove('training_batch_accuracy_top@1'), tags.remove('train_backward_timing')
        for tag in tags:
            x, y = [], []
            scalars[tag] = {}
            for scalar in event_acc.Scalars(tag):
                x.append(scalar.step)
                y.append(scalar.value)
            scalars[tag]['x'] = x
            scalars[tag]["y"] = y
        return scalars

    def _plot(self, x, y, title, label, xlabel="Epoch"):
        plt.xlabel(xlabel)
        plt.title(title)
        plt.plot(x, y, label=label)

    def plot_acc(self):
        for e in self.acc_list:
            title = e
            scalars = self.scalars[e]
            x, y = scalars["x"], scalars["y"]
            assert len(x) == len(y)
            self._plot(x, y, title, title)
            plt.legend(loc="best")
            self._plot_save(e)
            plt.clf()

    def _plot_save(self, title):
        fname = title + ".jpg"
        out_path = self.fig_dir.joinpath(fname)
        plt.savefig(out_path)

    def plot_acc_overlap(self):
        title_list = ["balanced_accuracy", "accuracy"]
        for title_target in title_list:
            for tag in self.acc_list:
                if "mean" in tag: continue
                title = "balanced_accuracy" if "bal" in tag else "accuracy"
                if title == title_target:
                    x, y, = self.scalars[tag]["x"], self.scalars[tag]["y"]
                    self._plot(x, y, title, tag)
            handles, labels = plt.gca().get_legend_handles_labels()
            handles, labels = self._sort_labels(handles, labels)
            plt.legend(handles, labels, loc="best")
            self._plot_save(title_target)
            plt.clf()

    def _sort_labels(self, handles, labels):
        sorted_labels = []
        for i, e in enumerate(labels):
            snr = e.split("dB")[0].split("_")[-1]
            sorted_labels.append((handles[i], (e, int(snr))))
        sorted_labels.sort(key=lambda x: x[1][1])
        sorted_labels = list(map(lambda x: (x[0], x[1][0]), sorted_labels))
        handles, sorted_labels = list(map(lambda x: x[0], sorted_labels)), list(map(lambda x: x[1], sorted_labels))
        return handles, sorted_labels

    def plot_loss(self):
        plt.clf()
        step = self.scalars["val_loss"]["x"][0]+1
        train_loss = self.scalars["train_loss"]
        x, y = train_loss["x"], train_loss["y"]
        x, y = x[::step], y[::step]
        epoch_list = list(range(len(x)))
        self._plot(epoch_list, y, title="", label="train_loss", xlabel="")
        val_loss = self.scalars["val_loss"]
        x, y = val_loss["x"], val_loss["y"]
        self._plot(epoch_list, y, title="", label="val_loss", xlabel="")
        plt.legend(loc="best")
        self._plot_save("loss")
        plt.clf()

    def plot_lr(self):
        plt.clf()
        lr = self.scalars["learning_rate"]
        x, y = lr["x"], lr["y"]
        self._plot(x, y, title="learning_rate", label="", xlabel="")
        self._plot_save("learning_rate")
        plt.clf()

    def plot_all(self):
        self.plot_loss()
        self.plot_acc()
        self.plot_acc_overlap()
        self.plot_lr()

    def save_confusion_matrix(self, figure: plt.Figure, snr):
        fname = f"confusion matrix {snr}dB.jpg"
        out = self.fig_dir.joinpath(fname)
        figure.set_size_inches(10, 10)
        figure.savefig(out, dpi=200, bbox_inches="tight")

# exp_dir = r"C:\MIE CARTELLE\PROGRAMMAZIONE\GITHUB\tesi_magistrale\nemo_experiments\MatchboxNet-3x2x64\2022-01-13_01-37-04"
# plot = PlotManager(exp_dir, 0, 30, 5)
# plot.plot_all()
