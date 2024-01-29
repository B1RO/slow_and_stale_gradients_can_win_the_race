This is an implementation of the K_Sync_SGD and K_Async_SGD algorithm.

The implementation uses a similar NN architecture as the paper "Slow and Stale Gradients Can Win The Race"[1], as it is not explicitly given.
To do a fair comparison, all other settings are the same, including a learning rate of 0.12, a batch size of 32 and a total of 8 worker nodes used.

The NN is trained and tested on the CIFAR-10 dataset.
The error-runtime performance is reported for a constant K of 2,4,8 as well as the adaptive AdaSync algorithm. 
Moreover, the final test accuracy is reported for the different variants.



References:
[1]. Sanghamitra Dutta, Jianyu Wang, Gauri Joshi. "Slow and Stale Gradients Can Win The Race". arXiv preprint arXiv:2003.10579. 2020



