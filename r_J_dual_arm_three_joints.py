import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from DualArm import DualArm
import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
Kp = 2
dt = 0.1
N_LINKS = 3  # Number of links in the arm
N_ITERATIONS = 10000
a_b = [2, 1]
# States
WAIT_FOR_NEW_GOAL = 1
MOVING_TO_GOAL = 2

def main():  # pragma: no cover
    """
    Creates a 3-link arm and uses its inverse kinematics to move it to the desired position.
    """
    link_lengths = [1, 1, 1]  # Lengths of the arm links
    joint_angles_a = np.array([0, 0, 0])
    joint_angles_b = np.array([0, 0, 0])  # Initial joint angles
    goal_pos_a = [3, 0]  # Desired goal position [x, y]
    goal_pos_b = [5, 0]  # Desired goal position [x, y]
    basic_point = [2, 0]
    show_animation = True

    dual_arm = DualArm(link_lengths, joint_angles_a, joint_angles_b, goal_pos_a, goal_pos_b, basic_point, a_b, show_animation)
    state_a = WAIT_FOR_NEW_GOAL
    solution_found_a = False
    state_b = WAIT_FOR_NEW_GOAL
    solution_found_b = False

    while True:
        old_goal_a = np.array(dual_arm.goal_a)
        old_goal_b = np.array(dual_arm.goal_b)
        dual_arm.update_joints(joint_angles_a, joint_angles_b)
        end_effector_a = np.array(dual_arm.end_effector_a)
        goal_pos_a = np.array(dual_arm.goal_a)
        errors_a, distance_a = distance_to_goal(end_effector_a, goal_pos_a)
        end_effector_b = np.array(dual_arm.end_effector_b)
        goal_pos_b = np.array(dual_arm.goal_b)
        errors_b, distance_b = distance_to_goal(end_effector_b, goal_pos_b)
        joint_angles = np.hstack((joint_angles_a,joint_angles_b))

        # State machine to allow changing of goal before current goal has been reached
        if state_a is WAIT_FOR_NEW_GOAL:
            if distance_a > 0.1 and not solution_found_a:
                joint_goal_angles, solution_found_a = inverse_kinematics_a(
                    link_lengths, joint_angles, goal_pos_a)
                joint_goal_angles_a = joint_goal_angles[:3]
                if not solution_found_a:
                    print("Solution of a could not be found.")
                    state_a = WAIT_FOR_NEW_GOAL
                    dual_arm.goal_a = end_effector_a
                elif solution_found_a:
                    state_a = MOVING_TO_GOAL
        elif state_a is MOVING_TO_GOAL:
            if distance_a > 0.1 and all(old_goal_a == goal_pos_a):
                joint_angles_a = joint_angles_a + Kp * \
                    ang_diff(joint_goal_angles_a, joint_angles_a) * dt
            else:
                state_a = WAIT_FOR_NEW_GOAL
                solution_found_a = False

        if state_b is WAIT_FOR_NEW_GOAL:
            if distance_b > 0.1 and not solution_found_b:
                joint_goal_angles, solution_found_b = inverse_kinematics_b(
                    link_lengths, joint_angles, goal_pos_b)
                joint_goal_angles_b = joint_goal_angles[3:]
                if not solution_found_b:
                    print("Solution of b could not be found.")
                    state_b = WAIT_FOR_NEW_GOAL
                    dual_arm.goal_b = end_effector_b
                elif solution_found_b:
                    state_b = MOVING_TO_GOAL
        elif state_b is MOVING_TO_GOAL:
            if distance_b > 0.1 and all(old_goal_b == goal_pos_b):
                joint_angles_b = joint_angles_b + Kp * \
                    ang_diff(joint_goal_angles_b, joint_angles_b) * dt
            else:
                state_b = WAIT_FOR_NEW_GOAL
                solution_found_b = False


        plt.clf()
        dual_arm.plot()
        plt.xlim(-3, 5)
        plt.ylim(-3, 3)
        plt.pause(0.0001)


def inverse_kinematics_a(link_lengths, joint_angles, goal_pos):
    """
    Calculates the inverse kinematics using the Jacobian inverse method.
    """
    for iteration in range(N_ITERATIONS):
        current_pos_a = forward_kinematics(link_lengths, joint_angles[:3])
        errors, distance = distance_to_goal(current_pos_a, goal_pos)
        if distance < 0.1:
            print("Solution found in %d iterations." % iteration)
            return joint_angles, True
        J = r_jacobian_inverse(link_lengths, joint_angles[:3], joint_angles[3:])
        joint_angles = joint_angles + np.matmul(J, errors)
    return joint_angles, False

def inverse_kinematics_b(link_lengths, joint_angles, goal_pos):
    """
    Calculates the inverse kinematics using the Jacobian inverse method.
    """
    for iteration in range(N_ITERATIONS):
        current_pos_b = forward_kinematics(link_lengths, joint_angles[3:])
        errors, distance = distance_to_goal(current_pos_b, goal_pos)
        if distance < 0.1:
            print("Solution found in %d iterations." % iteration)
            return joint_angles, True
        J = r_jacobian_inverse(link_lengths, joint_angles[:3], joint_angles[3:])
        joint_angles = joint_angles + np.matmul(J, errors)
    return joint_angles, False
def forward_kinematics(link_lengths, joint_angles):
    x = y = 0
    for i in range(1, N_LINKS + 1):
        x += link_lengths[i - 1] * np.cos(np.sum(joint_angles[:i]))
        y += link_lengths[i - 1] * np.sin(np.sum(joint_angles[:i]))
    return np.array([x, y]).T

def distance_to_goal(current_pos, goal_pos):
    x_diff = goal_pos[0] - current_pos[0]
    y_diff = goal_pos[1] - current_pos[1]
    return np.array([x_diff, y_diff]).T, np.hypot(x_diff, y_diff)


def ang_diff(theta1, theta2):
    """
    Returns the difference between two angles in the range -pi to +pi
    """
    return (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi



def r_jacobian_inverse(link_lengths, joint_angles_a, joint_angles_b):
    r_J = np.zeros((2, 2 * N_LINKS))
    J_a = np.zeros((2, N_LINKS))
    J_b = np.zeros((2,N_LINKS))
    for i in range(N_LINKS):
        J_a[0, i] = 0
        J_a[1, i] = 0
        for j in range(i, N_LINKS):
            J_a[0, i] -= link_lengths[j] * np.sin(np.sum(joint_angles_a[:j]))
            J_a[1, i] += link_lengths[j] * np.cos(np.sum(joint_angles_a[:j]))

    for i in range(N_LINKS):
        J_b[0, i] = 0
        J_b[1, i] = 0
        for j in range(i, N_LINKS):
            J_b[0, i] -= link_lengths[j] * np.sin(np.sum(joint_angles_b[:j]))
            J_b[1, i] += link_lengths[j] * np.cos(np.sum(joint_angles_b[:j]))

    theta_a = np.sum(joint_angles_a)
    theta_b = np.sum(joint_angles_b)
    rotation_a_o = np.array([[np.cos(theta_a), -np.sin(theta_a)],
                                [np.sin(theta_a), np.cos(theta_a)]])
    rotation_b_o = np.array([[np.cos(theta_b), -np.sin(theta_b)],
                                [np.sin(theta_b), np.cos(theta_b)]])
    r_j1 = - (np.matmul(rotation_a_o.T , J_a))
    r_j2 = np.matmul(np.matmul(rotation_a_o.T , rotation_b_o.T) , J_b)
    r_J = np.hstack((r_j1, r_j2))
    return np.linalg.pinv(r_J)


if __name__ == '__main__':
    main()