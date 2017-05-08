# 2d-fem
Python code to solve for the displacement of a 2d beam using Finite Element Method. 
Built as part of a Design Project on FEM under Dr. Gaurav Singh (Department of Mechanical Engineering, BITS Pilani Goa Campus).
January to May 2017

Specifications:
- The mesh is rectangular and can be of arbitrary fineness.
- Inputs are beam geometry, physical parameters (Young's modulus and Poisson's ratio), body forces (such as weight) and traction forces (externally applied)
- Output is an array of node-by-node displacements in equilibrium, and a visual plot of the same
- Successive Over Relaxation is used to find the final displacements, though NumPy's numpy.linalg.solve performs similarly well in time and accuracy.
