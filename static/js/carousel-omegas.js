function generateHTML(labelNumber) {
    return `
    <div class="carousel results-carousel">
      <div class="item">
        <img src="static/images/omegas_inpainting/sample_label${labelNumber}_cond0.png"/>
        <h2 class="subtitle has-text-centered">
          &omega;=0.0
        </h2>
      </div>
      <div class="item">
        <img src="static/images/omegas_inpainting/sample_label${labelNumber}_cond1.png"/>
        <h2 class="subtitle has-text-centered">
          &omega;=1.5
        </h2>
      </div>
      <div class="item">
        <img src="static/images/omegas_inpainting/sample_label${labelNumber}_cond3.png"/>
        <h2 class="subtitle has-text-centered">
          &omega;=3.0
        </h2>
      </div>
      <div class="item">
        <img src="static/images/omegas_inpainting/sample_label${labelNumber}_cond4.png"/>
        <h2 class="subtitle has-text-centered">
          &omega;=4.5
        </h2>
      </div>
      <div class="item">
        <img src="static/images/omegas_inpainting/sample_label${labelNumber}_cond6.png"/>
        <h2 class="subtitle has-text-centered">
          &omega;=6.0
        </h2>
      </div>
    </div>`
  }

  document.addEventListener('DOMContentLoaded', function() {
    for(let i = 1; i <= 8; i++) {
      const containerA = document.getElementById(`carousel-container-label${i}a`);
      containerA.innerHTML = generateHTML(i);
      const containerB = document.getElementById(`carousel-container-label${i}b`);
      containerB.innerHTML = generateHTML(i);
    }
  });