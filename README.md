


<div align="center">
  <a href="(https://github.com/panliangrui/FORESEE/blob/main/frame.png)">
    <img src="https://github.com/panliangrui/FORESEE/blob/main/frame.png" width="900" height="400" />
  </a>

  <h1>Flowchart depicting multimodal patient survival prediction.</h1>

  <p>
  Liangrui Pan et al. is a developer helper.
  </p>

  <p>
    <a href="https://github.com/misitebao/yakia/blob/main/LICENSE">
      <img alt="GitHub" src="https://img.shields.io/github/license/misitebao/yakia"/>
    </a>
  </p>

  <!-- <p>
    <a href="#">Installation</a> | 
    <a href="#">Documentation</a> | 
    <a href="#">Twitter</a> | 
    <a href="https://discord.gg/zRC5BfDhEu">Discord</a>
  </p> -->

  <div>
  <strong>
  <samp>

[English](README.md)

  </samp>
  </strong>
  </div>
</div>

# FORESEE: Multimodal and Multi-view Representation Learning for Robust Prediction of Cancer Survival

## Table of Contents

<details>
  <summary>Click me to Open/Close the directory listing</summary>

- [Table of Contents](#table-of-contents)
- [Feature Preprocessing](#Feature-Preprocessing)
  - [Feature Extraction](#Feature-Extraction)
  - [Graph Construction](#Graph-Construction)
- [Train Models](#Train-models)
- [Datastes](#Datastes)
- [Website](#Website)
- [License](#license)

</details>

## Feature Preprocessing

Use the pre-trained model for feature preprocessing and build the spatial topology of WSI.
The relevant clinical data and histopathological image data can be downloaded at https://www.cbioportal.org/ and https://portal.gdc.cancer.gov/.

### Feature Extraction

Features extracted based on KimiaNet.
Please refer to KimiaNet: https://github.com/KimiaLabMayo/KimiaNet.
```markdown
cd cut_and_pretrain
python new_cut7.py
```

### Graph Construction

Use KNN (K=8) to construct the spatial topology map.
```markdown
cd Graph
python construct_graph.py
```

## Train Models
```markdown
python train.py
python train_abl.py
python train_HAE_abl.py
python train_mae_abl.py
```

## Datastes

- We provide the relevant features of histopathology images, the download link is as follows:
```markdown
  link：https://pan.baidu.com/s/1pJY1Cv9d-ML7jU09RnGOjg?pwd=rzj7
  https://zenodo.org/records/11611418

```

## License
The code will be updated after the paper is accepted!!
[License MIT](../LICENSE)
