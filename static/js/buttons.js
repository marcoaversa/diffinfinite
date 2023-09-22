function showImage(imageId) {
    // Remove 'active' class from all images
    document.querySelectorAll('.image').forEach(img => {
      img.classList.remove('active');
    });
    
    // Add 'active' class to the selected image
    document.getElementById(imageId).classList.add('active');
}

function showCarousel(carouselId) {
    // First, hide all carousels
    document.querySelectorAll('.carousel').forEach(carousel => {
        carousel.style.display = 'none';
    });

    // Now, show the chosen carousel
    document.getElementById(carouselId).style.display = 'block';
}

document.addEventListener('DOMContentLoaded', function() {
// Initialize carousels here
document.getElementById('carousel1').innerHTML = generateHTML(1);
document.getElementById('carousel2').innerHTML = generateHTML(2);
});