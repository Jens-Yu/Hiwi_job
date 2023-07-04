import matplotlib.pyplot as plt
import numpy as np

# Simulation parameters
Kp = 2
dt = 0.1

# Link lengths
l1 = 1.0
l2 = 1.0
l3 = 1.0

# Set initial goal position to the initial end-effector position
x = 3.0
y = 0.0

show_animation = True

if show_animation:
    plt.ion()


def three_joint_arm(GOAL_TH=0.0, theta1=0.0, theta2=0.0, theta3=0.0):
    """
    Computes the inverse kinematics for a planar 3DOF arm
    When out of bounds, rewrite x and y with last correct values
    """
    global x, y
    x_prev, y_prev = None, None
    while True:
        try:
            if x is not None and y is not None:
                x_prev = x
                y_prev = y
            if np.hypot(x, y) > (l1 + l2 + l3):
                theta3_goal = 0
            else:
                goal = np.array([x_prev, y_prev])
                jacobian = np.zeros((2, 3))
                jacobian[0, 0] = -l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2) - l3 * np.sin(theta1 + theta2 + theta3)
                jacobian[0, 1] = -l2 * np.sin(theta1 + theta2) - l3 * np.sin(theta1 + theta2 + theta3)
                jacobian[0, 2] = -l3 * np.sin(theta1 + theta2 + theta3)
                jacobian[1, 0] = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2) + l3 * np.cos(theta1 + theta2 + theta3)
                jacobian[1, 1] = l2 * np.cos(theta1 + theta2) + l3 * np.cos(theta1 + theta2 + theta3)
                jacobian[1, 2] = l3 * np.cos(theta1 + theta2 + theta3)
                theta_goal = np.dot(np.linalg.pinv(jacobian), goal)

            theta1_goal = theta_goal[0]
            theta2_goal = theta_goal[1]
            theta3_goal = theta_goal[2]

            theta1 = theta1 + Kp * ang_diff(theta1_goal, theta1) * dt
            theta2 = theta2 + Kp * ang_diff(theta2_goal, theta2) * dt
            theta3 = theta3 + Kp * ang_diff(theta3_goal, theta3) * dt
        except ValueError as e:
            print("Unreachable goal" + str(e))
        except TypeError:
            x = x_prev
            y = y_prev

        wrist = plot_arm(theta1, theta2, theta3, x, y)

        # check goal
        d2goal = None
        if x is not None and y is not None:
            d2goal = np.hypot(wrist[0] - x, wrist[1] - y)

        if abs(d2goal) < GOAL_TH and x is not None:
            return theta1, theta2, theta3


def plot_arm(theta1, theta2, theta3, target_x, target_y):
    shoulder = np.array([0, 0])
    elbow = shoulder + np.array([l1 * np.cos(theta1), l1 * np.sin(theta1)])
    wrist = elbow + np.array([l2 * np.cos(theta1 + theta2), l2 * np.sin(theta1 + theta2)])
    end_effector = wrist + np.array([l3 * np.cos(theta1 + theta2 + theta3), l3 * np.sin(theta1 + theta2 + theta3)])

    if show_animation:
        plt.cla()

        plt.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], 'k-')
        plt.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], 'k-')
        plt.plot([wrist[0], end_effector[0]], [wrist[1], end_effector[1]], 'k-')

        plt.plot(shoulder[0], shoulder[1], 'ro')
        plt.plot(elbow[0], elbow[1], 'ro')
        plt.plot(wrist[0], wrist[1], 'ro')
        plt.plot(end_effector[0], end_effector[1], 'ro')

        plt.plot([end_effector[0], target_x], [end_effector[1], target_y], 'g--')
        plt.plot(target_x, target_y, 'g*')

        plt.xlim(-3, 3)
        plt.ylim(-3, 3)

        plt.show()
        plt.pause(dt)

    return end_effector


def ang_diff(theta1, theta2):
    # Returns the difference between two angles in the range -pi to +pi
    return (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi


def click(event):
    global x, y
    x = event.xdata
    y = event.ydata


def main():
    fig = plt.figure()
    fig.canvas.mpl_connect("button_press_event", click)
    fig.canvas.mpl_connect('key_release_event', lambda event: [
        exit(0) if event.key == 'escape' else None])
    three_joint_arm()


if __name__ == "__main__":
    main()