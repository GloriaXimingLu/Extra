
MOVE_AHEAD = 'MoveAhead'
ROTATE_LEFT = 'RotateLeft'
ROTATE_RIGHT = 'RotateRight'
LOOK_UP = 'LookUp'
LOOK_DOWN = 'LookDown'
#DONE = 'Done'
OpenObject = 'OpenObject'
CloseObject = 'CloseObject'
PickupObject = 'PickupObject'
PlaceHeldObject = 'PlaceHeldObject'

BASIC_ACTIONS = [MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, LOOK_UP, LOOK_DOWN, OpenObject, CloseObject, PickupObject, PlaceHeldObject]

GOAL_SUCCESS_REWARD = 5
STEP_PENALTY = -0.01
FAILED_ACTION_PENALTY = -1
PROCESS_REWARD = 1