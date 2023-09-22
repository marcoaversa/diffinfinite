function showCarousel(carouselId) {
    // First, hide all carousels
    document.querySelectorAll('.grid').forEach(grid => {
        grid.style.display = 'none';
    });

    // Now, show the chosen carousel
    document.getElementById(carouselId).style.display = 'block';
}