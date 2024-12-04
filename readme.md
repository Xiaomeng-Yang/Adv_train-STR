# Accurate and Robust Scene Text Recognition via Adversarial Training
This repository is the implementation of paper "[Accurate and Robust Scene Text Recognition via Adversarial Training](https://ieeexplore.ieee.org/abstract/document/10445827)".

## Dependency
- This work was tested with PyTorch1.3.1 , CUDA 10.2, python 3.6 and Ubuntu 18.04.
- This work is based on the [advtorch](https://github.com/BorealisAI/advertorch/) and [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).
- All of the requirements are listed in requirements.txt

## AT and Evaluation for robustness
- Train our best model for CRNN-AT:

```bash
CUDA_VISIBLE_DEVICES=0 python train_adv_regularization.py --train_data PATH_TO_TRAINING_DATASET --valid_data PATH_TO_VALIDATION_DATASET --Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC --pgd_eps 0.03 --pgd_nb_iter 1 --alpha 0.6
```

- Evaluate model robustness against attacks:

```bash
CUDA_VISIBLE_DEVICES=0 python test_robustness.py --eval_data PATH_TO_EVALUATION_DATASET --benchmark_all_eval --Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC --saved_model PATH_TO_MODEL --attack (LinfPGD/L2PGD/FGSM) --attack_eps 0.03 --pgd_nb_iter 1
```

## Reference
[1] Baek, J.; Kim, G.; Lee, J.; Park, S.; Han, D.; Yun, S.; Oh, S. J.; and Lee, H. 2019. What is wrong with scene text recognition model comparisons? Dataset and Model Analysis. In ICCV.

[2] Ding, G. W.; Wang, L.; and Jin, X. 2019.  AdverTorch v0.1: An adversarial robustness toolbox based on PyTorch. arXiv preprint arXiv:1902.07623

