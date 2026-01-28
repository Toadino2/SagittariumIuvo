These Python files, along with the C++ code provided once compiled to an executable, can be interpreted to produce a Kivy-based user application where archers
can input their equipment details and sessions as sequences of volleys and arrow positions, as well as metadata, save them as files and perform different statistical analyses of choice on them. Most
classical descriptive statistics are supported; most importantly, it is possible to perform inference to check sight correctness, characterize the data-generating process
and therefore the session itself, and use unsupervised learning (via clustering) to detect possible classes of mistakes. More elementary inference on scores and coordinate variances
is also supported, as well as techniques to detect possible faulty arrows; much of the inference can also be applied to the corresponding polar coordinates. The application allows users
to visualize the analysis and to tweak parameters. Finally, general-level analysis of the datasets is provided as a time series, allowing for progress tracking, performance prediction
and supervised learning to measure the impact of different equipment setups or environmental conditions.

The application will be expanded in the future: a focus is on finding methods that allow to more precisely single out shooting mistakes and on experimenting with methods with more
relaxed assumptions or increased efficiency and speed.
