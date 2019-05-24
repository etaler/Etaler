This file documents the steps to add GUI features to Etaler running under container.
* on the host run "xhost local:root" to enable X11 connection on localhost from the container
* Enable ETALER_ENABLE_MATPLOTLIB_CPP in the examples directory cmake list file
* build the examples