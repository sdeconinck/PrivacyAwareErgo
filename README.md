# Source code for the paper Enabling Privacy-Aware AI-Based Ergonomic Analysis

To be completed

## Prepare dataset

The `train_obfuscator.py` script expects dataset in the yolo format (indicated by a .yaml file, more info on the [ultralytics documentation](https://docs.ultralytics.com/datasets/pose/#ultralytics-yolo-format)).

For the dataset used in the paper, you can use the script `prepare_dataset.sh`. It expects the input data in the form of:

*   `input_folder/`
    *   `train/`
        *   `scene1/`
            *   `vid1/`
            *   `vid2/`
            *   `vid3/`
            *   `vid4/`
        *   `scene2/`
        *   `...`
    *   `val/`
        *   `...`
    *   `test/`
        *   `...`

Your config should look like this then:

```yaml
path: path_to_output_folder_given_in_script
train: images/train
val: images/val
test: images/test

kpt_shape: [17,3]

names:
  0: person
```

## Train Obfuscator/Deobfuscator

To train an obfuscator/deobfuscator model, use the `train_obfuscator.py` file. This script expects a number of input arguments, use `python3 train_obfuscator.py --help` for more info.

For a base variant you can use:

```bash
python3 train_obfuscator.py --config_path your_config_file_here --nowandb
```
