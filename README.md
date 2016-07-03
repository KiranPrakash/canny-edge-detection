# canny-edge-detection
In 1986, John Canny defined a set of goals for an edge detector and described an optimal method for achieving them. Canny specified three issues that an edge detector must address: 1. Error rate: Desired edge detection filter should find all the edges, there should not be any missing edges, and it should respond only to edge regions. 2. Localization: Distance between detected edges and actual edges should be as small as possible. 3. Response: The edge detector should not identify multiple edge pixels where only a single edge exists. In the following scripts, we will first smooth the images, then compute gradients, magnitude, and orientation of the gradient. This procedure is followed by non-max suppression, and finally hysteresis thresholding is applied to finalize the steps.


![swan] (https://raw.githubusercontent.com/KiranPrakash/canny-edge-detection/master/Berkley%20Dataset/43070.jpg)
