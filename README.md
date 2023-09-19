# DiffInfinite

<div style="text-align: center;">We present DiffInfinite, a hierarchical diffusion model that generates arbitrarily large histological images while preserving long-range correlation structural information. Our approach first generates synthetic segmentation masks, subsequently used as conditions for the high-fidelity generative diffusion process. The proposed sampling method can be scaled up to any desired image size while only requiring small patches for fast training. Moreover, it can be parallelized more efficiently than previous large-content generation methods while avoiding tiling artefacts. The training leverages classifier-free guidance to augment a small, sparsely annotated dataset with unlabelled data. Our method alleviates unique challenges in histopathological imaging practice: large-scale information, costly manual annotation, and protective data handling. The biological plausibility of DiffInfinite data is validated in a survey by ten experienced pathologists as well as a downstream segmentation task. Furthermore, the model scores strongly on anti-copying metrics which is beneficial for the protection of patient data.</div>

![Example Image](images/examples/synth_examples.png)

<div style="text-align: center;">Infinite Generation Sampling Method:</div>

<div style="text-align: center;">
  <video width="320" height="240" controls>
    <source src="video.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
<!-- ![Video](images/video.m4v) -->
<\div>

Click on the link below to run the Jupyter Notebook on Google Colab (set ```colab=True``` in the first cell):

<a target="_blank" href="https://colab.research.google.com/github/diffinfinite/diffinfinite/blob/master/main.ipynb">

  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
