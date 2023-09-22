function showCarousel(carouselId) {
    // Hide all images
    var carousels = document.getElementsByClassName('container');
    for(var i = 0; i < carousels.length; i++) {
        carousels[i].style.display = 'none';
    }
    
    // Show the selected image
    document.getElementById(carouselId).style.display = 'block';
  }