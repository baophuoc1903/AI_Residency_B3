from matplotlib import pyplot as plt


class Logger:

    def __init__(self):
        self.loss_train = []
        self.loss_val = []

    def get_logs(self):
        return self.loss_train, self.loss_val

    def restore_logs(self, logs):
        self.loss_train, self.loss_val = logs

    def add_logs(self, loss_train, loss_val):
        self.loss_train.append(loss_train)
        self.loss_val.append(loss_val)

    def visualize(self):
        plt.plot(self.loss_train, 'g', label='Training Loss')
        plt.plot(self.loss_val, 'b', label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
