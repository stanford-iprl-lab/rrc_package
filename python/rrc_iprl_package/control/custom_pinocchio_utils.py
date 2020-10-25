import numpy as np

import pinocchio

from trifinger_simulation.pinocchio_utils import Kinematics

class CustomPinocchioUtils(Kinematics):
  """
  Consists of kinematic methods for the finger platform.
  """

  def __init__(self, finger_urdf_path, tip_link_names):
    """
    Initializes the finger model on which control's to be performed.

    Args:
        finger (SimFinger): An instance of the SimFinger class
    """
    super().__init__(finger_urdf_path, tip_link_names)
 
  def get_tip_link_jacobian(self, finger_id, q):
    """
    Get Jacobian for tip link of specified finger
    All other columns are 0
    """
    pinocchio.computeJointJacobians(
        self.robot_model, self.data, q,
    )
    #pinocchio.framesKinematics(
    #    self.robot_model, self.data, q,
    #)
    pinocchio.framesForwardKinematics(
        self.robot_model, self.data, q,
    )
    frame_id = self.tip_link_ids[finger_id]
    Ji = pinocchio.getFrameJacobian(
      self.robot_model,
      self.data,
      frame_id,
      pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )

    #print(self.robot_model.frames[frame_id].placement)
    #print(self.data.oMf[frame_id].rotation)
    return Ji

