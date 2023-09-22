function showCarousel(carouselId) {
    const carousels = document.getElementsByClassName('container');
    for (let carousel of carousels) {
        carousel.style.display = carousel.id === carouselId ? 'block' : 'none';
    }
}
