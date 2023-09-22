function showCarousel(carouselId) {
    // Hide all images
    var carousels = document.getElementsByClassName('selected-image');
    for(var i = 0; i < carousels.length; i++) {
        carousels[i].style.display = 'none';
    }

    document.getElementById(carouselId).style.display = 'block';
  }

  document.addEventListener("DOMContentLoaded", function() {
    setTimeout(function(){
        document.getElementById('Artery').style.display = 'none';
        document.getElementById('Artifacts').style.display = 'none';
        document.getElementById('Carcinoma').style.display = 'none';
        document.getElementById('Cartilage').style.display = 'none';
        document.getElementById('Connective').style.display = 'none';
        document.getElementById('Necrosis').style.display = 'none';
        document.getElementById('Stroma').style.display = 'none';
    }, 1); // 1000 milliseconds = 1 second
  });