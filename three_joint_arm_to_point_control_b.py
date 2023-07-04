import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from NLinkArm_b import NLinkArm_b

# Simulation parameters
Kp = 2
dt = 0.1
N_LINKS = 3  # Number of links in the arm
N_ITERATIONS = 10000

# States
WAIT_FOR_NEW_GOAL = 1
MOVING_TO_GOAL = 2

show_animation = True


def main():  # pragma: no cover
    """
    Creates a 3-link arm and uses its inverse kinematics to move it to the desired position.
    """
    link_lengths = [1, 1, 1]  # Lengths of the arm links
    joint_angles = np.array([0, 0, 0])  # Initial joint angles
    goal_pos = [5, 0]  # Desired goal position [x, y]
    basic_point=[2 ,0]
    arm = NLinkArm_b(link_lengths, joint_angles, goal_pos, show_animation,basic_point)
    state = WAIT_FOR_NEW_GOAL
    solution_found = False

    while True:
        old_goal = np.array(goal_pos)
        goal_pos = np.array(arm.goal)
        end_effector = arm.end_effector
        errors, distance = distance_to_goal(end_effector, goal_pos)

        # State machine to allow changing of goal before current goal has been reached
        if state is WAIT_FOR_NEW_GOAL:
            if distance > 0.1 and not solution_found:
                joint_goal_angles, solution_found = inverse_kinematics(
                    link_lengths, joint_angles, goal_pos)
                if not solution_found:
                    print("Solution could not be found.")
                    state = WAIT_FOR_NEW_GOAL
                    arm.goal = end_effector
                elif solution_found:
                    state = MOVING_TO_GOAL
        elif state is MOVING_TO_GOAL:
            if distance > 0.1 and all(old_goal == goal_pos):
                joint_angles = joint_angles + Kp * \
                    ang_diff(joint_goal_angles, joint_angles) * dt
            else:
                state = WAIT_FOR_NEW_GOAL
                solution_found = False

        arm.update_joints(joint_angles)


def inverse_kinematics(link_lengths, joint_angles, goal_pos):
    """
    Calculates the inverse kinematics using the Jacobian inverse method.
    """
    for iteration in range(N_ITERATIONS):
        current_pos = forward_kinematics(link_lengths, joint_angles)
        errors, distance = distance_to_goal(current_pos, goal_pos)
        if distance < 0.1:
            print("Solution found in %d iterations." % iteration)
            return joint_angles, True
        J = jacobian_inverse(link_lengths, joint_angles)
        joint_angles = joint_angles + np.matmul(J, errors)
    return joint_angles, False


def forward_kinematics(link_lengths, joint_angles):
    x = y = 0
    for i in range(1, N_LINKS + 1):
        x += link_lengths[i - 1] * np.cos(np.sum(joint_angles[:i]))
        y += link_lengths[i - 1] * np.sin(np.sum(joint_angles[:i]))
    return np.array([x, y]).T


def jacobian_inverse(link_lengths, joint_angles):
    J = np.zeros((2, N_LINKS))
    for i in range(N_LINKS):
        J[0, i] = 0
        J[1, i] = 0
        for j in range(i, N_LINKS):
            J[0, i] -= link_lengths[j] * np.sin(np.sum(joint_angles[:j]))
            J[1, i] += link_lengths[j] * np.cos(np.sum(joint_angles[:j]))

    return np.linalg.pinv(J)


def distance_to_goal(current_pos, goal_pos):
    x_diff = goal_pos[0] - current_pos[0]
    y_diff = goal_pos[1] - current_pos[1]
    return np.array([x_diff, y_diff]).T, np.hypot(x_diff, y_diff)


def ang_diff(theta1, theta2):
    """
    Returns the difference between two angles in the range -pi to +pi
    """
    return (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi


if __name__ == '__main__':
    main()