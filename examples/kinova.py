from math import *
import numpy as np
from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.arm_models import KinovaRobotTemplate

class KinovaRobot(KinovaRobotTemplate):
    def __init__(self):
        super().__init__()
    
    def dh_to_H(self,dh_table):
        H_list = []
        for i in range(dh_table.shape[0]):
            theta = dh_table[i,0]
            d = dh_table[i,1]
            a = dh_table[i,2]
            alpha = dh_table[i,3]
            H_list.append(np.array([
                [cos(theta),-sin(theta)*cos(alpha),sin(theta)*sin(alpha),a*cos(theta)],
                [sin(theta),cos(theta)*cos(alpha),-cos(theta)*sin(alpha),a*sin(theta)],
                [0,sin(alpha),cos(alpha),d],
                [0,0,0,1]
            ]))

            if (i == 0):
                H_ee = H_list[0]
            else:
                # build on previous H
                H_ee = H_ee @ H_list[i]
        
        return H_ee, H_list

    def calc_forward_kinematics(self,joint_values: list, radians=True):
        dh_table = np.array([
            [0,self.l1,0,pi],
            [self.joint_values[0],-self.l2, 0, 0.5*pi],
            [self.joint_values[1] + 0.5*pi, 0, -self.l3,pi],
            [self.joint_values[2] + 0.5*pi, 0, 0, 0.5*pi],
            [self.joint_values[3], -self.l4 - self.l5, 0, -0.5*pi],
            [self.joint_values[4], 0, 0, 0.5*pi],
            [self.joint_values[5], -self.l6 - self.l7, 0, pi]
        ])

        H_ee, H_list = self.dh_to_H(dh_table=dh_table)

        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_ee @ np.array([0, 0, 0, 1]))[:3]

        rpy = ut.rotm_to_euler(H_ee[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, H_list
    
    def calc_inverse_kinematics(self, ee, joint_values, soln):
        """
        Calculates analytical inverse kinematics for the joint
        """
        # We have 3 choices for solutions
        opt1  = (soln >> 2) & 1
        opt2 = (soln >> 1) & 1
        opt3 = soln & 1

        R_0_ee = ut.euler_to_rotm((ee.rotx,ee.roty,ee.rotz))
        p_ee = np.array([ee.x,ee.y,ee.z])
        z_ee = R_0_ee @ np.array([0,0,1]) # Z axis of end effector
        p_wrist = p_ee - (self.l6 + self.l7)*z_ee # Bend in wrist position
        print(f"p_ee: {p_ee}, p_wrist: {p_wrist}")

        # The joints before the wrist consist only of the two DOF arm on a pivot
        if opt3:
            theta1 = np.atan2(p_wrist[1],p_wrist[0])
        else:
            theta1 = np.atan2(p_wrist[1],p_wrist[0]) + pi
        #print(f"theta1: {theta1}") # theta 1 seems good

        # # Undoes rotation of initial joint
        initial_rotation = np.array([[cos(theta1),-sin(theta1),0],
                                    [sin(theta1),cos(theta1),0],
                                    [0,0,1]])
        
        # Rotate p_wrist so it's in the XZ plane (undoing theta1 rotation)
        p_wrist_transformed = initial_rotation.T @ p_wrist

        # Shift p_wrist to correspond to translation from joint 1
        p_wrist_transformed[2] = p_wrist_transformed[2] - (self.l1 + self.l2)
        cosB = (-p_wrist_transformed[0]**2 - p_wrist_transformed[2]**2 + self.l3**2 + (self.l4+self.l5)**2)/(2*self.l3*(self.l4+self.l5))
        #print(f"cos b is {cosB}")
        #cosB = np.clip(cosB, -1.0, 1.0)
        beta = np.acos(cosB)

        if opt2:
            theta3 = pi - beta
        else:
            theta3 = beta - pi
        
        #print(f"p wrist transformed: {p_wrist_transformed}")
        
        #alpha = np.atan2((self.l4+self.l5)*sin(beta),self.l3 + (self.l4+self.l5)*cos(beta))
        alpha = np.asin(((self.l4 + self.l5) * np.sin(beta))/((p_wrist_transformed[0]**2 + p_wrist_transformed[2]**2))**0.5)
        gam = np.atan2(p_wrist_transformed[2],p_wrist_transformed[0])

        if opt2:
            theta2 = gam - alpha
        else:
            theta2 = gam + alpha
        
        print(f"Thetas 1,2,3: {theta1}, {theta2}, {theta3}")

        # Get the orientation of the 3rd frame (wrist) w.r.t. base frame
        dh_table_partial = np.array([
            [0,0,0,pi],
            [theta1,-self.l2-self.l1, 0, 0.5*pi],
            [theta2 + 0.5*pi, 0, -self.l3,pi],
            [theta3 + 0.5*pi, 0, 0, 0.5*pi]
        ])

        H_0_3,_ = self.dh_to_H(dh_table_partial)
        R_0_3 = H_0_3[0:3,0:3]

        # We want rotation of EE w.r.t. frame 3 (before wrist)
        R_3_ee = R_0_3.T @ R_0_ee
        
        if opt1:
            theta5 = np.acos(-R_3_ee[2,2])
            theta4 = np.atan2(R_3_ee[1,2],R_3_ee[0,2])
            theta6 = np.atan2(R_3_ee[2,1],R_3_ee[2,0])
        else:
            theta5 = -np.acos(-R_3_ee[2,2])
            theta4 = np.atan2(-R_3_ee[1,2],-R_3_ee[0,2])
            theta6 = np.atan2(-R_3_ee[2,1],-R_3_ee[2,0])

        print(f"Returned angles: {[theta1,theta2,theta3,theta4,theta5,theta6]}")
        return [theta1,theta2,theta3,theta4,theta5,theta6]
    

    def calc_numerical_ik(self, ee, joint_values, tol=0.0001, ilimit=100):
        # Numerical IK
        n = ilimit
        eps = tol
        attempts = 10

        p_des = np.array([ee.x,ee.y,ee.z,ee.rotx,ee.roty,ee.rotz])
        for j in range(attempts):
            # Random guess
            curr_joint_vals = ut.sample_valid_joints(self)
            for i in range(n):
                print(f"Curr joint vals: {curr_joint_vals}")
                p_ee, _ = self.calc_forward_kinematics(curr_joint_vals)
                err = p_des - np.array([p_ee.x, p_ee.y, p_ee.z, p_ee.rotx, p_ee.roty, p_ee.rotz]) # error vector (position)
                print(f"Error {np.linalg.norm(err)}")

                if np.linalg.norm(err) < eps:
                    print(f"Found solution at iter {i} with error {np.linalg.norm(err)}")
                    return curr_joint_vals
                
                J = self.calc_jacobians(curr_joint_vals)
                # We only care about the 2x2 Jacobian (2 joints, 2 pose parameters we control)
                J = J[0:6,0:6]
                # Damped inverse
                lam = 0.000
                JJt = J @ J.T
                JJt = JJt + (lam**2 * np.eye(6))
                J_inv = J.T @ np.linalg.pinv(JJt)
                step = J_inv @ err

                curr_joint_vals = curr_joint_vals + step
                #curr_joint_vals = [ut.wraptopi(val) for val in curr_joint_vals]
                if not ut.check_joint_limits(curr_joint_vals, self.joint_limits):
                    break
                    
        print("Failed to converge")
        return None
    
    def calc_jacobians(self,joint_values):
        """
        Calculates Jacobian matrix given joint angles
        """
        # Jacobian will be 6 rows by 5 colums
        k_hat = np.array([0,0,1])
        J = np.zeros(shape=(6,6))

        # Get ee position and H transform matrices
        ee, H_list = self.calc_forward_kinematics(joint_values=joint_values,radians=True)

        pos_ee = np.array([ee.x, ee.y, ee.z])

        H_0_current = np.eye(4)
        for i,H in enumerate(H_list):
            R_i = H_0_current[0:3,0:3]
            d_i = H_0_current[0:3,3]
            
            pos_joint_to_ee = pos_ee - d_i # lever arm of joint to ee
            z_joint = R_i @ k_hat # axis of rotation of joint

            Jv = np.cross(z_joint,pos_joint_to_ee) # cross product is jacobian
            Jw = z_joint

            #print(f"Jv: {Jv}, Jw: {Jw}")

            J[0:3,i] = Jv
            J[3:6,i] = Jw

            H_0_current = H_0_current @ H
        
        #print(f"\nMy Jacobian was: {J}\n")
        return J


if __name__ == "__main__":
    model = KinovaRobot()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()