This Git has the main files used in this robotic liver biopsy project. the folders names trial[number] contain the environment and custom weights for eight obstacle-avoidance trajectories. Please note that trial 1 has all the code files (agent, network, buffer, testSAC, train_sac) and the files are thoroughly commented. Please refer back to my dissertation for a more detailed undertstanding of the technical aspects. the rest of the folders are not commented. To implement, run train_sac after modifying accordingly (number of weights, replay buffer etc)

the needle MATLAB files are the inverse kinematics for the stage 1 robot. please implement FABRIKc and then casecompute1 for complete IK. 

The arduino files can be used to run the motors for the stage 1 and stage 2 robots. 

