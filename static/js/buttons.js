function showCarousel(carouselId) {
    // Hide all images
    var carousels = document.getElementsByClassName('grid');
    for(var i = 0; i < carousels.length; i++) {
        carousels[i].style.display = 'none';
    }
    
    // Show the selected image
    document.getElementById(carouselId).style.display = 'block';
  }