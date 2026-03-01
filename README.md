# Native-Binarization
Preventing Collapse in 1-Bit Diffusion

---

To use, first, install the `model.architectures.py` as package using:

```pip install -e .```

---

Now you can do whatever you want haha. Do note that:

- All important codes(Bencmarker, Quantizer and Trainer) are inside `code`
- I am not very smart
- `data` contains MNIST dataset of 60K-ish images ig (about 63MB)
- Pre-trained models are in `pre_trained_models`
- Has Apache 2.0 License
- Model architectures are defined inside `.\models\architectures.py` (containing: FP16, W1A16, W1A1 and an MNIST Classifier)
- `models` folder has `bnn_w1a1_16batch.pth` and `bnn_w1a1_128batch.pth`, which are **W1A1** models trained in 16 batche_size and 128 batch_size respectively
- Please make sure to use CUDA otherwise this will destroy your lifetime
- This DOES NOT have a MNIST Classifier trainer (I think I accidentally deleted it). That's why I have this amazing placeholder at `.\code\Trainers\mnist_trainer.py`. (If I get time, I'll make it, or if possible, you're most definitely welcome to contribute!!)
- That `model_output_generator.py` is used to do a 3-way comparison between an FP16, W1A16 and W1A1 model.

---

Now, there are some tiny simple naming conventions I used, so that it's easy-to-remember:
- In the `pre_trained_models` folder,
    - fp16_to_w1a1: An FP16 quantized to W1A1
    - W1A1: A W1A1 trained using Our Method(ie trained from scratch with pre-activation)
    - fp16_to_w1a16: An FP16 quantized to W1A16
    - W1A16: A W1A16 trained using Our Method

- The Architecture File has 4 Classes, namely
    - ResUNet_FP16
    - ResUNet_W1A16
    - ResuNet_W1A1
    - MNISTClassifier


