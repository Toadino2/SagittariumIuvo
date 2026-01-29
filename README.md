# Archery Performance Analysis App

This repository contains a prototype application for the statistical analysis
of archery training sessions, developed in Python with performance-critical
components in C++ and Stan, and a Kivy-based graphical interface.

The application allows users to:
- input equipment details and training sessions as sequences of volleys and arrow positions
- store sessions and metadata (equipment, conditions, etc.)
- perform a range of statistical analyses on positional, angular, and score data
- visualize results interactively and adjust analysis parameters

### Implemented analyses
The current prototype supports:
- classical descriptive statistics on scores and coordinates
- statistical inference to assess sight correctness and session characteristics
- analysis of coordinate and polar representations
- detection of potential faulty arrows
- unsupervised learning techniques (clustering) to identify possible classes of shooting mistakes
- time series analysis of sessions for progress tracking and performance prediction
- supervised learning experiments to evaluate the impact of equipment and environmental conditions

### Development status
The project is a functional prototype focused on core logic, statistical methodology,
and overall architecture. The current codebase prioritizes extensibility and experimentation.
Further testing, optimization, and feature expansion are planned. 
The statistical analysis core has been extensively exercised in practice,
while parts of the GUI and orchestration layer are currently under active development
and may require further refinement; also, the statistical core has been recently reorganized for better integration with the orchestrating code
but testing is still underway (previous versions of it still however work as intended). Full integration of a computer vision model is also
planned (see https://universe.roboflow.com/statisticallearningcolbrutti/statisticallearningcolbrutti).


### Repository contents
- Python code for data handling, analysis, and GUI
- C++ / Stan components compiled into Python-callable binaries
- example datasets
- setup scripts for building native extensions

