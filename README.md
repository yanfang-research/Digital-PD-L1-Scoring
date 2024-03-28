# Digital PD-L1 Scoring in DLBCL

## Artificial Intelligence-based Assessment of PD-L1 Expression in Diffuse Large B Cell Lymphoma

[Journal Link](https://www.nature.com/articles/s41698-024-00577-y#citeas) | [Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41698-024-00577-y/MediaObjects/41698_2024_577_MOESM2_ESM.pdf)

This repository provides the official implementation of paper titled "Artificial Intelligence-based Assessment of PD-L1 Expression in Diffuse Large B Cell Lymphoma" published in npj Precision Oncology.

**Abstract:** Diffuse large B cell lymphoma (DLBCL) is an aggressive blood cancer known for its rapid progression and high incidence. The growing use of immunohistochemistry (IHC) has significantly contributed to the detailed cell characterization, thereby playing a crucial role in guiding treatment strategies for DLBCL. In this study, we developed an AI-based image analysis approach for assessing PD-L1 expression in DLBCL patients. PD-L1 expression represents as a major biomarker for screening patients who can benefit from targeted immunotherapy interventions. In particular, we performed large-scale cell annotations in IHC slides, encompassing over 5101 tissue regions and 146,439 live cells. Extensive experiments in primary and validation cohorts demonstrated the defined quantitative rule helped overcome the difficulty of identifying specific cell types. In assessing data obtained from fine needle biopsies, experiments revealed that there was a higher level of agreement in the quantitative results between Artificial Intelligence (AI) algorithms and pathologists, as well as among pathologists themselves, in comparison to the data obtained from surgical specimens. We highlight that the AI-enabled analytics enhance the objectivity and interpretability of PD-L1 quantification to improve the targeted immunotherapy development in DLBCL patients.

<!-- Insert the project banner here -->
<div align="center">
    <a href="https://"><img width="1000px" height="auto" src="https://github.com/yanfang-research/Digital-PD-L1-Scoring/blob/main/WSI_TPS.jpg"></a>
</div>

---

<!-- give a introduction of your project -->

## 🛠️ Get Started

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
- Step 1: ROI Segmentation
- Step 2: Cell Detection
- Step 3: Cell Segmentation
- Step 4: TPS Analysis

## 🛡️ License

This project is under the MIT license. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgement

We are deeply grateful to the contributors of the following papers and softwares.

- [Vision Transformer (ViT)](https://github.com/google-research/vision_transformer)
- [Nuclick](https://github.com/navidstuv/NuClick)
- [AUXCNN](https://github.com/shenghh2015/cell-counting)
- [ASAP](https://computationalpathologygroup.github.io/ASAP/)
- [LabelMe](https://github.com/labelmeai/labelme)

## 📝 Citation

If you find our work useful in your research or if you use parts of this code please consider citing our [paper](https://www.nature.com/articles/s41698-024-00577-y#citeas):

Yan, F., Da, Q., Yi, H. et al. Artificial intelligence-based assessment of PD-L1 expression in diffuse large B cell lymphoma. npj Precis. Onc. 8, 76 (2024). https://doi.org/10.1038/s41698-024-00577-y
        
        
        
        

```
@article{chen2024uni,
  title={Artificial intelligence-based assessment of PD-L1 expression in diffuse large B cell lymphoma},
  author={Fang Yan, Qian Da, Hongmei Yi, Shijie Deng, Lifeng Zhu, Mu Zhou, Yingting Liu, Ming Feng, Jing Wang, Xuan Wang, Yuxiu Zhang, Wenjing Zhang, Xiaofan Zhang, Jingsheng Lin, Shaoting Zhang & Chaofu Wang},
  journal={npj Precision Oncology},
  publisher={Springer Nature},
  year={2024}
}
```
