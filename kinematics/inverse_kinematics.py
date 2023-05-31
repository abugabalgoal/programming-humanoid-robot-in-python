'''In this exercise you need to implement inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
'''


from forward_kinematics import ForwardKinematicsAgent
from numpy.matlib import identity
import numpy as np


class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''
        joint_angles = []
        lambda_ = 0.001

        for chain in self.chains:
            for joint in self.chains[chain]:
                joint_angles[joint] = self.perception.joint[joint]

        target = np.array([self.from_trans(transform)]).T

        for i in range(3000):
            self.forward_kinematics(joint_angles)

            T = [0] * len(self.chains[effector_name])
            for i, name in enumerate(self.chains[effector_name]):
                T[i] = self.transforms[name]

            Te = np.array([self.from_trans(T[-1])])
            e = target - Te
            T = np.array([self.from_trans(i) for i in T[0:len(self.chains[effector_name])]])
            J = (Te - T).T
            J[-1, :] = 1
            d_theta = lambda_ * np.dot(np.dot(J.T, np.linalg.pinv(np.dot(J, J.T))), e.T)

            for i, name in enumerate(self.chains[effector_name]):
                joint_angles[name] += np.asarray(d_theta.T)[0][i]

            if np.linalg.norm(d_theta) < 1e-4:
                break
        
        
        return joint_angles

    def set_transforms(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # YOUR CODE HERE
        angles = self.inverse_kinematics(effector_name,transform)
        names = []
        times = []
        keys = []
        for name in self.chains[effector_name]:
            if name == 'LWristYaw' or name == 'RWristYaw':
                continue
            names.append(name)
            times.append([2, 5])
            keys.append([[self.perception.joint[name], [3, 0, 0], [3, 0, 0]], [angles[name], [3, 0, 0], [3, 0, 0]]])
        self.keyframes = (names, times, keys)

if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = identity(4)
    T[-1, 1] = 0.05
    T[-1, 2] = -0.26
    agent.set_transforms('LLeg', T)
    agent.run()
