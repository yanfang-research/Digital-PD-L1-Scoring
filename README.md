# Artificial Intelligence-based Assessment of PD-L1 Expression in Diffuse Large B Cell Lymphoma

This repository provides the official implementation of paper titled "Artificial Intelligence-based Assessment of PD-L1 Expression in Diffuse Large B Cell Lymphoma".

<!-- Select some of the point info, feel free to delete -->
🧙 Updated on 2024.01.24. Paper will be coming soon!

<!-- Insert the project banner here -->
<div align="center">
    <a href="https://"><img width="800px" height="auto" src="https://github.com/yanfang-research/Digital-PD-L1-Scoring/blob/main/WSI_TPS.jpg"></a>
</div>

---

## Links
We are deeply grateful to the contributors of the following papers and softwares.

- [Vision Transformer (ViT)](https://github.com/google-research/vision_transformer)
- [Nuclick](https://github.com/navidstuv/NuClick)
- [AUXCNN](https://github.com/shenghh2015/cell-counting)
- [ASAP](https://computationalpathologygroup.github.io/ASAP/)
- [LabelMe](https://github.com/labelmeai/labelme)

<!-- give a introduction of your project -->

## Get Started

**Main Requirements**  
> Python==3.9
> 
> torch==1.8
> 
> Openslide==3.4
>
> Numpy==1.16
>
> Pandas==0.25
>
> opencv-python==4.6
> 

**Pipeline**

The code is modified from [MoCo v3](https://github.com/facebookresearch/moco-v3).

For basic MoCo v3 training, 
```python
python main_moco.py \
  --tcga ./used_TCGA.csv \
  -a vit_base -b 2048 --workers 128 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=100 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --dist-backend nccl \
  --dist-url 'tcp://localhost:10001' \
  [your dataset folders]
```


## 🛡️ License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgement

- Shanghai AI Laboratory.
- Shanghai Ruijin Hospital.

## 📝 Citation

If you find this repository useful, please consider citing our paper.
```
Coming soon...
