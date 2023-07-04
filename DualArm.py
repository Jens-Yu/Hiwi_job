import numpy as np
import matplotlib.pyplot as plt


class DualArm(object):
    def __init__(self, link_lengths, joint_angles_a, joint_angles_b, goal_a, goal_b, basic_point, show_animation):
        self.show_animation = show_animation
        self.n_links = len(link_lengths)
        if self.n_links != len(joint_angles_a) or self.n_links != len(joint_angles_b):
            raise ValueError()

        self.link_lengths = np.array(link_lengths)
        self.lim = sum(link_lengths)

        self.joint_angles_a = np.array(joint_angles_a)
        self.points_a = [[0, 0] for _ in range(self.n_links + 1)]
        self.goal_a = np.array(goal_a).T

        self.joint_angles_b = np.array(joint_angles_b)
        self.basic_point = basic_point
        self.points_b = [[basic_point[0],basic_point[1]] for _ in range(self.n_links + 1)]
        self.goal_b = np.array(goal_b).T

        if show_animation:  # pragma: no cover
            self.fig = plt.figure()
            self.fig.canvas.mpl_connect('button_press_event', self.click)

            plt.ion()
            plt.show()

        self.update_points()

    def update_joints(self, joint_angles_a, joint_angles_b):
        self.joint_angles_a = joint_angles_a
        self.joint_angles_b = joint_angles_b
        self.update_points()

    def update_points(self):
        for i in range(1, self.n_links + 1):
            self.points_a[i][0] = self.points_a[i - 1][0] + \
                self.link_lengths[i - 1] * \
                np.cos(np.sum(self.joint_angles_a[:i]))
            self.points_a[i][1] = self.points_a[i - 1][1] + \
                self.link_lengths[i - 1] * \
                np.sin(np.sum(self.joint_angles_a[:i]))
            self.points_b[i][0] = self.points_b[i - 1][0] + \
                self.link_lengths[i - 1] * \
                np.cos(np.sum(self.joint_angles_b[:i]))
            self.points_b[i][1] = self.points_b[i - 1][1] + \
                self.link_lengths[i - 1] * \
                np.sin(np.sum(self.joint_angles_b[:i]))

        self.end_effector_a = np.array(self.points_a[self.n_links]).T
        self.end_effector_b = np.array(self.points_b[self.n_links]).T

        if self.show_animation:  # pragma: no cover
            self.plot()

    def plot(self):  # pragma: no cover
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

        for i in range(self.n_links + 1):
            if i is not self.n_links:
                plt.plot([self.points_a[i][0], self.points_a[i + 1][0]],
                         [self.points_a[i][1], self.points_a[i + 1][1]], 'r-')
                plt.plot([self.points_b[i][0], self.points_b[i + 1][0]],
                         [self.points_b[i][1], self.points_b[i + 1][1]], 'r-')
            plt.plot(self.points_a[i][0], self.points_a[i][1], 'ko')
            plt.plot(self.points_b[i][0], self.points_b[i][1], 'ko')
        plt.plot(self.goal_a[0], self.goal_a[1], 'gx')
        plt.plot(self.goal_b[0] + self.basic_point[0], self.goal_b[1] + self.basic_point[1], 'gx')

        plt.plot([self.end_effector_a[0], self.goal_a[0]], [
                 self.end_effector_a[1], self.goal_a[1]], 'g--')
        plt.plot([self.end_effector_b[0], self.goal_b[0] + self.basic_point[0]], [
                 self.end_effector_b[1], self.goal_b[1] + self.basic_point[1]], 'g--')

        plt.xlim([-self.lim, self.lim + self.basic_point[0]])
        plt.ylim([-self.lim, self.lim])
        plt.draw()
        plt.pause(0.0001)

    def click(self, event):
        if event.button == 1:
            self.goal_a = np.array([event.xdata, event.ydata]).T
        if event.button == 3:
            self.goal_b = np.array([event.xdata - self.basic_point[0], event.ydata - self.basic_point[1]]).T
        self.plot()


