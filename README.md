---
layout: page
title: DiffInfinite: Large Mask-Image Synthesis via Parallel Random Patch Diffusion in Histopathology
---

<!-- Center-align the title -->
<center>
  <h1>{{ page.title }}</h1>
</center>

<!-- Center-align the authors in a paper format -->
<div class="paper">
  <h3>Authors:</h3>
  <ul>
    <li>Author 1</li>
    <li>Author 2</li>
    <li>Author 3</li>
  </ul>
</div>

<!-- Create boxes with links to the code and paper -->
<div class="paper">
  <h3>Links:</h3>
  <ul>
    <li><a href="https://github.com/your-username/your-repo">Code</a></li>
    <li><a href="https://arxiv.org/your-paper-link">Paper</a></li>
  </ul>
</div>


<!-- Add the abstract -->
<h2>Abstract</h2>
<p>We present DiffInfinite, a hierarchical diffusion model that generates arbitrarily large histological images while preserving long-range correlation structural information. Our approach first generates synthetic segmentation masks, subsequently used as conditions for the high-fidelity generative diffusion process. The proposed sampling method can be scaled up to any desired image size while only requiring small patches for fast training. Moreover, it can be parallelized more efficiently than previous large-content generation methods while avoiding tiling artefacts. The training leverages classifier-free guidance to augment a small, sparsely annotated dataset with unlabelled data. Our method alleviates unique challenges in histopathological imaging practice: large-scale information, costly manual annotation, and protective data handling. The biological plausibility of DiffInfinite data is validated in a survey by ten experienced pathologists as well as a downstream segmentation task. Furthermore, the model scores strongly on anti-copying metrics which is beneficial for the protection of patient data.</p>

<!-- Display an image -->
<div class="paper">
  <h3>Image:</h3>
  <img src="images/examples/synth_examples.png">
</div>

<!-- Embed a video (replace 'your-video.mp4' with your video file) -->
<div class="paper">
  <h3>Video:</h3>
  <video width="640" height="360" controls>
    <source src="images/video.m4v" type="video/m4v">
    Your browser does not support the video tag.
  </video>
</div>

Click on the link below to run the Jupyter Notebook on Google Colab (set ```colab=True``` in the first cell):

<a target="_blank" href="https://colab.research.google.com/github/diffinfinite/diffinfinite/blob/master/main.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


We present DiffInfinite, a hierarchical diffusion model that generates arbitrarily large histological images while preserving long-range correlation structural information. Our approach first generates synthetic segmentation masks, subsequently used as conditions for the high-fidelity generative diffusion process. The proposed sampling method can be scaled up to any desired image size while only requiring small patches for fast training. Moreover, it can be parallelized more efficiently than previous large-content generation methods while avoiding tiling artefacts. The training leverages classifier-free guidance to augment a small, sparsely annotated dataset with unlabelled data. Our method alleviates unique challenges in histopathological imaging practice: large-scale information, costly manual annotation, and protective data handling. The biological plausibility of DiffInfinite data is validated in a survey by ten experienced pathologists as well as a downstream segmentation task. Furthermore, the model scores strongly on anti-copying metrics which is beneficial for the protection of patient data.

![Example Image](images/examples/synth_examples.png)

<div style="text-align: center;">Infinite Generation Sampling Method:</div>

<div style="text-align: center;">
    <video width="960" height="720" controls>
        <source src="video.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>
<!-- ![Video](images/video.m4v) -->

Click on the link below to run the Jupyter Notebook on Google Colab (set ```colab=True``` in the first cell):

<a target="_blank" href="https://colab.research.google.com/github/diffinfinite/diffinfinite/blob/master/main.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
