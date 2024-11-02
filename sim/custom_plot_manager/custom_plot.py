import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt


class PltManager:
    def __init__(self):
        self.data = {}

    def reset(self):
        self.data = {}

    def log(self, name, value):
        if name not in self.data:
            self.data[name] = []
        self.data[name].append(value)

    def plot(self, save_fig=True, save_data=True, log_dir=None):
        figures = {}
        # 解析数据键以适应不同的层级
        for name, values in self.data.items():
            paths = name.split("/")
            if len(paths) == 1:
                fig_title = paths[0]
                subplot_title = "Main"  # 默认子图标题
                line_title = "Data"  # 默认曲线标题
            elif len(paths) == 2:
                fig_title, subplot_title = paths
                line_title = "Data"  # 默认曲线标题
            elif len(paths) == 3:
                fig_title, subplot_title, line_title = paths
            else:
                raise ValueError("Log entry cannot have more than two '/' characters.")

            if fig_title not in figures:
                figures[fig_title] = {}
            if subplot_title not in figures[fig_title]:
                figures[fig_title][subplot_title] = {}
            figures[fig_title][subplot_title][line_title] = values

        # save data
        if save_data:
            file_name = osp.join(log_dir, "custom_plot", "data.txt")
            os.makedirs(osp.dirname(file_name), exist_ok=True)
            with open(file_name, "w") as f:
                for name, values in self.data.items():
                    f.write(f"{name}: {values}\n")

        # 绘制图表和子图
        for fig_title, subplots in figures.items():
            n = len(subplots)
            if n % 4 == 0:
                n_row = n // 4
                n_col = 4
            elif n % 3 == 0:
                n_row = n // 3
                n_col = 3
            else:
                n_row = n
                n_col = 1
            fig, axs = plt.subplots(
                n_row, n_col, figsize=(n_col * 4, n_col * 3), squeeze=False
            )
            fig.suptitle(fig_title, fontsize=16)

            for ax, (subplot_title, lines) in zip(axs.flatten(), subplots.items()):
                for line_title, line_values in lines.items():
                    ax.plot(line_values, label=line_title)
                ax.set_title(subplot_title)
                ax.legend()

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            if save_fig:
                file_name = osp.join(log_dir, "custom_plot", f"{fig_title}.png")
                os.makedirs(osp.dirname(file_name), exist_ok=True)
                plt.savefig(file_name)

            # plt.show()

    def close(self):
        plt.close("all")
