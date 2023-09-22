function showImage(imageId) {
    // Remove 'active' class from all images
    document.querySelectorAll('.image').forEach(img => {
      img.classList.remove('active');
    });
    
    // Add 'active' class to the selected image
    document.getElementById(imageId).classList.add('active');
  }